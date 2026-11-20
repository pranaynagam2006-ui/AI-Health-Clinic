import json
import re
import sqlite3
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from app.db import Database
from app.errors import (
    ApprovalConflictError,
    InventoryAgentError,
    InvalidToolArgumentsError,
    LLMQuotaExceededError,
    LLMUnavailableError,
)
from app.llm import structured_planner
from app.retrieval import PolicyRetriever
from app.schemas import ApprovalDecision, PlannerDecision
from app.tools import TOOL_SCHEMAS, ToolRegistry


class AgentState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    user_input: str
    active_tool_calls: list[dict[str, Any]]
    execution_steps: list[dict[str, Any]]
    tool_calls_completed: int
    thread_id: str
    response: str
    order_id: str | None
    order_total: float | None
    approval_required: bool
    approval_payload: dict[str, Any]
    policy_context: list[str]
    planning_mode: str


TOOL_DESCRIPTIONS = {
    "query_inventory": (
        "Look up medical supplies by SKU, text, category, or clinic; low_stock_only filters to "
        "items at/below reorder point. All arguments optional: sku?, search?, category?, clinic_id?, low_stock_only?."
    ),
    "query_purchase_orders": (
        "Look up supplier purchase orders, including pending status, vendor, clinic, and line items. "
        "Arguments are optional: status?, vendor_name?, clinic_id?."
    ),
    "query_vendors": "Look up supplier lead times, ratings, and minimum quantities. Argument: vendor_name?.",
    "calculate_reorder_point": (
        "Calculate lead-time demand and whether stock is at/below the reorder threshold. "
        "Arguments: stock_level, daily_usage, lead_time_days."
    ),
    "create_draft_purchase_order": (
        "Create a supplier purchase-order draft. Quantity is raised to supplier MOQ if necessary. "
        "Arguments: sku, quantity, clinic_id? (defaults to product's assigned clinic)."
    ),
}


def summarize_tool_result(tool_name: str, result: Any, arguments: dict[str, Any]) -> str:
    if tool_name == "query_inventory":
        records = result if isinstance(result, list) else []
        if not records:
            return "I couldn't find any inventory items matching that request."
        clinic_names = sorted({row.get("clinic_name") or row.get("clinic_id") or "an unassigned clinic" for row in records})
        low_stock = [
            row for row in records
            if row.get("stock_level") is not None
            and row.get("reorder_point") is not None
            and row["stock_level"] <= row["reorder_point"]
        ]
        item_label = "item" if len(records) == 1 else "items"
        clinic_label = "clinic" if len(clinic_names) == 1 else "clinics"
        if arguments.get("low_stock_only"):
            message = (
                f"I found {len(records)} low-stock supply {item_label} across "
                f"{len(clinic_names)} {clinic_label}."
            )
        else:
            message = (
                f"I found {len(records)} supply {item_label} across "
                f"{len(clinic_names)} {clinic_label}; "
                f"{len(low_stock)} are at or below their reorder points."
            )
        examples = [
            f"{row.get('product_name', row.get('sku', 'Supply'))} at "
            f"{row.get('clinic_name') or row.get('clinic_id') or 'an unassigned clinic'} "
            f"({row.get('stock_level')} in stock, reorder point {row.get('reorder_point')})"
            for row in records[:3]
        ]
        example_label = "For example, " if len(examples) == 1 else "Examples: "
        return f"{message} {example_label}{'; '.join(examples)}."

    if tool_name == "query_vendors":
        vendors = result if isinstance(result, list) else []
        if not vendors:
            return "I couldn't find any suppliers matching that request."
        lead_times = [vendor["lead_time_days"] for vendor in vendors if vendor.get("lead_time_days") is not None]
        minimums = [
            vendor["minimum_order_quantity"]
            for vendor in vendors
            if vendor.get("minimum_order_quantity") is not None
        ]
        if lead_times and minimums:
            return (
                f"I found {len(vendors)} supplier{'s' if len(vendors) != 1 else ''}. Lead times range from "
                f"{min(lead_times)} to {max(lead_times)} days, and minimum order quantities range from "
                f"{min(minimums)} to {max(minimums)} units."
            )
        return f"I found {len(vendors)} supplier{'s' if len(vendors) != 1 else ''}."

    if tool_name == "query_purchase_orders":
        orders = result if isinstance(result, list) else []
        if not orders:
            return "I couldn't find any purchase orders matching that request."
        pending = sum(order.get("status") == "pending" for order in orders)
        total_cents = sum(order.get("total_cost_cents", 0) for order in orders)
        total_text = f" Their combined value is ${total_cents / 100:,.2f}." if total_cents else ""
        pending_text = f", including {pending} awaiting approval" if pending else ""
        return (
            f"I found {len(orders)} purchase order{'s' if len(orders) != 1 else ''}"
            f"{pending_text}.{total_text}"
        )

    if tool_name == "calculate_reorder_point" and isinstance(result, dict):
        status = "Reordering is recommended." if result.get("needs_reorder") else "Current stock is above the reorder point."
        return (
            f"Based on usage and supplier lead time, the reorder point is {result['reorder_point']} units; "
            f"current stock is {result['stock_level']} units. {status}"
        )

    return "The requested inventory action completed."


def build_graph(
    database: Database,
    llm: Any,
    retriever: PolicyRetriever,
    checkpointer: Any | None = None,
    approval_threshold: float = 0,
    llm_error: InventoryAgentError | None = None,
):
    registry = ToolRegistry(database)

    def deterministic_read_plan(message: str) -> PlannerDecision | None:
        normalized = message.casefold()
        if re.search(r"\b(create|draft|place|submit|prepare)\b.{0,40}\b(order|purchase order)\b", normalized):
            return None
        if (
            "low stock" in normalized
            or "low-stock" in normalized
            or "supplies needing reorder" in normalized
            or "items to reorder" in normalized
        ):
            return PlannerDecision(
                action="tool_call",
                tool_call={"name": "query_inventory", "arguments": {"low_stock_only": True}},
            )
        if re.search(r"\bppe\b", normalized):
            return PlannerDecision(
                action="tool_call",
                tool_call={"name": "query_inventory", "arguments": {"category": "PPE"}},
            )
        if any(term in normalized for term in ("supplier terms", "vendor terms", "lead time", "minimum order")):
            return PlannerDecision(
                action="tool_call",
                tool_call={"name": "query_vendors", "arguments": {}},
            )
        if "pending purchase" in normalized or "pending order" in normalized:
            return PlannerDecision(
                action="tool_call",
                tool_call={"name": "query_purchase_orders", "arguments": {"status": "pending"}},
            )
        clinic_id = re.search(r"\bclinic[- ]?(\d{3})\b", normalized)
        if clinic_id:
            return PlannerDecision(
                action="tool_call",
                tool_call={
                    "name": "query_inventory",
                    "arguments": {"clinic_id": f"CLINIC-{clinic_id.group(1)}"},
                },
            )
        if "lakeshore" in normalized:
            return PlannerDecision(
                action="tool_call",
                tool_call={"name": "query_inventory", "arguments": {"clinic_id": "CLINIC-003"}},
            )
        return None

    def append_step(state: AgentState, step: dict[str, Any]) -> list[dict[str, Any]]:
        return [*state.get("execution_steps", []), step]

    def planner_agent(state: AgentState) -> dict[str, Any]:
        thread_id = state["thread_id"]
        user_input = state["user_input"]
        started = time.perf_counter()
        database.log_event(
            thread_id,
            "planner",
            "step_started",
            {"step": "plan_request"},
        )
        decision = deterministic_read_plan(user_input)
        planning_mode = "deterministic_read" if decision else "llm"
        policy_context: list[str] = []
        if decision:
            intent = decision.tool_call.name if decision.tool_call is not None else "respond"
            database.log_event(
                thread_id,
                "planner",
                "intent_classified",
                {"mode": "deterministic_read", "intent": intent},
            )
        else:
            policy_context = retriever.search(user_input)
            database.log_event(
                thread_id,
                "planner",
                "policy_retrieval",
                {"documents_found": len(policy_context)},
            )
        tools_text = "\n".join(f"- {name}: {description}" for name, description in TOOL_DESCRIPTIONS.items())
        prompt = (
            "You are the Planner Agent for a local inventory and supplier-purchasing workflow. "
            "Select at most one available tool or respond directly. Never invent inventory, prices, "
            "clinic, or supplier details. "
            "Never invent purchase status. Never write, request, or execute SQL. There are no INSERT, "
            "UPDATE, or DELETE tools. "
            "A purchase order must only be drafted when explicitly requested by the user. "
            "Use tool results already present in the conversation as authoritative. You may make "
            "at most three tool calls in this request. Use the supplied policy excerpts when relevant. "
            "Tool catalog:\n"
            f"{tools_text}\n\nSupplier and purchasing policy excerpts:\n"
            f"{json.dumps(policy_context, ensure_ascii=True)}"
        )
        if decision is None:
            if llm_error is not None:
                raise llm_error
            try:
                planner = structured_planner(llm)
                decision = planner.invoke(
                    [SystemMessage(content=prompt), *state.get("messages", [HumanMessage(content=user_input)])]
                )
                decision = PlannerDecision.model_validate(decision)
            except (InvalidToolArgumentsError, LLMUnavailableError):
                raise
            except ValidationError as exc:
                database.log_event(
                    thread_id,
                    "planner",
                    "tool_arguments_rejected",
                    {
                        "valid": False,
                        "validation_errors": [
                            {"location": error["loc"], "type": error["type"], "message": error["msg"]}
                            for error in exc.errors()
                        ],
                    },
                )
                raise InvalidToolArgumentsError(f"Planner returned invalid tool arguments: {exc}") from exc
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                if "429" in error_text or "quota" in error_text.lower() or "resourceexhausted" in error_text.lower():
                    raise LLMQuotaExceededError(
                        "Gemini free-tier quota is exhausted or temporarily limited. "
                        "Wait for the displayed retry period or switch LLM_PROVIDER=ollama."
                    ) from exc
                raise LLMUnavailableError(f"Planner Agent could not obtain a valid plan: {exc}") from exc

        calls = [decision.tool_call.model_dump()] if decision.tool_call else []
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        if decision.tool_call:
            database.log_event(
                thread_id,
                "planner",
                "tool_arguments_validated",
                {
                    "tool": decision.tool_call.name,
                    "schema": TOOL_SCHEMAS[decision.tool_call.name].__name__,
                    "valid": True,
                },
            )
        database.log_event(
            thread_id,
            "planner",
            "decision",
            {
                "action": decision.action,
                "tool_call": calls[0] if calls else None,
                "duration_ms": duration_ms,
                "planning_mode": planning_mode,
            },
        )
        steps = append_step(
            state,
            {
                "agent": "planner",
                "action": decision.action,
                "tool_call": calls[0] if calls else None,
                "argument_validation": "passed" if calls else None,
                "duration_ms": duration_ms,
                "planning_mode": planning_mode,
            },
        )
        return {
            "active_tool_calls": calls,
            "execution_steps": steps,
            "policy_context": policy_context,
            "planning_mode": planning_mode,
            "response": decision.response,
            "messages": [AIMessage(content=decision.response or f"Planned action: {decision.action}")],
        }

    def inventory_agent(state: AgentState) -> dict[str, Any]:
        call = state["active_tool_calls"][0]
        thread_id = state["thread_id"]
        started = time.perf_counter()
        database.log_event(
            thread_id,
            "inventory",
            "step_started",
            {"tool": call["name"], "argument_keys": sorted(call["arguments"])},
        )
        result = registry.execute(call["name"], call["arguments"], thread_id=thread_id)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        database.log_event(
            thread_id,
            "inventory",
            "tool_call",
            {
                "tool": call["name"],
                "arguments": call["arguments"],
                "result": result,
                "argument_validation": "passed",
                "duration_ms": duration_ms,
            },
        )
        order_id = result.get("order_id") if isinstance(result, dict) else None
        order_total = result.get("total_cost") if isinstance(result, dict) else None
        is_order = call["name"] == "create_draft_purchase_order"
        if is_order and order_id:
            response = (
                f"Draft {order_id}: {result['quantity']} x {result['product_name']} from "
                f"{result['supplier_name']} for ${result['total_cost']:.2f}."
            )
        else:
            response = summarize_tool_result(call["name"], result, call["arguments"])
        tool_result_message = json.dumps(result, ensure_ascii=True, default=str)
        step = {
            "agent": "inventory",
            "tool": call["name"],
            "arguments": call["arguments"],
            "result": result,
            "argument_validation": "passed",
            "duration_ms": duration_ms,
        }
        return {
            "active_tool_calls": [],
            "execution_steps": append_step(state, step),
            "tool_calls_completed": state.get("tool_calls_completed", 0) + 1,
            "order_id": order_id,
            "order_total": order_total,
            "response": response,
            "messages": [AIMessage(content=tool_result_message)],
        }

    def approval_guardrail_agent(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        order_id = state.get("order_id")
        if not order_id:
            return {}
        order_total = float(state.get("order_total") or 0)
        thread_id = state["thread_id"]
        database.log_event(
            thread_id,
            "approval_guardrail",
            "step_started",
            {"order_id": order_id, "total": order_total},
        )
        if order_total <= approval_threshold:
            database.set_order_status(order_id, "approved")
            database.log_event(
                thread_id,
                "approval_guardrail",
                "auto_approved",
                {
                    "order_id": order_id,
                    "total": order_total,
                    "threshold": approval_threshold,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            return {
                "approval_required": False,
                "response": f"{state.get('response', '')} This order is below the approval threshold and was auto-approved.",
            }

        # This write is idempotent because an interrupted LangGraph node is re-entered on resume.
        if database.record_pending_approval(order_id, thread_id):
            database.log_event(
                thread_id,
                "approval_guardrail",
                "approval_requested",
                {
                    "order_id": order_id,
                    "total": order_total,
                    "threshold": approval_threshold,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
        decision_value = interrupt(
            {
                "type": "purchase_order_approval",
                "order_id": order_id,
                "total": order_total,
                "message": "Approve this supplier purchase order? Respond YES or NO.",
            }
        )
        try:
            decision = ApprovalDecision.model_validate(
                decision_value if isinstance(decision_value, dict) else {"decision": decision_value}
            )
        except Exception as exc:
            raise InvalidToolArgumentsError(f"Approval must be YES or NO: {exc}") from exc
        try:
            database.resolve_approval(order_id, thread_id, decision.decision)
        except ValueError as exc:
            raise ApprovalConflictError(str(exc)) from exc
        database.log_event(
            thread_id,
            "approval_guardrail",
            "approval_decision",
            {
                "order_id": order_id,
                "decision": decision.decision,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        status_text = "approved" if decision.decision == "YES" else "rejected"
        response = f"Purchase order {order_id} was {status_text}."
        return {
            "approval_required": True,
            "approval_payload": {"order_id": order_id, "decision": decision.decision},
            "response": response,
            "messages": [AIMessage(content=response)],
        }

    def route_after_planning(state: AgentState) -> str:
        return "inventory" if state.get("active_tool_calls") else END

    def route_after_inventory(state: AgentState) -> str:
        if state.get("order_id"):
            return "approval_guardrail"
        if state.get("planning_mode") == "deterministic_read":
            return END
        if state.get("tool_calls_completed", 0) < 3:
            return "planner"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_agent)
    workflow.add_node("inventory", inventory_agent)
    workflow.add_node("approval_guardrail", approval_guardrail_agent)
    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges("planner", route_after_planning, {"inventory": "inventory", END: END})
    workflow.add_conditional_edges(
        "inventory",
        route_after_inventory,
        {"planner": "planner", "approval_guardrail": "approval_guardrail", END: END},
    )
    workflow.add_edge("approval_guardrail", END)
    if checkpointer is None:
        checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def sqlite_checkpointer(path: str) -> Any:
    """Create a durable LangGraph SQLite checkpointer using Python's sqlite3 connection."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    connection = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()
    return saver


def invoke_chat(graph: Any, message: str, thread_id: str) -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(
        {
            "messages": [HumanMessage(content=message)],
            "user_input": message,
            "active_tool_calls": [],
            "execution_steps": [],
            "tool_calls_completed": 0,
            "thread_id": thread_id,
        },
        config=config,
    )


def resume_approval(graph: Any, thread_id: str, decision: str) -> dict[str, Any]:
    return graph.invoke(
        Command(resume={"decision": decision}),
        config={"configurable": {"thread_id": thread_id}},
    )
