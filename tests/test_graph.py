from langgraph.checkpoint.memory import MemorySaver
from pydantic import ValidationError
import pytest

from app.config import Settings
from app.db import Database
from app.errors import LLMUnavailableError, MissingAPIKeyError
from app.graph import build_graph, invoke_chat, resume_approval, sqlite_checkpointer
from app.llm import create_llm
from app.retrieval import PolicyRetriever
from app.schemas import PlannerDecision, ToolCall


class FixedPlanner:
    def __init__(self, decision):
        self.decision = decision

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        return self.decision


def seeded_database(tmp_path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    database.upsert_inventory(
        sku="GLOVE-M",
        product_name="Gloves medium",
        stock_level=3,
        daily_usage=2,
        lead_time_days=4,
        reorder_point=8,
        supplier_name="Supply Co",
        supplier_email="orders@example.test",
        minimum_order_quantity=10,
        unit_cost_cents=125,
    )
    return database


def test_approval_interrupt_persists_and_resumes(tmp_path):
    database = seeded_database(tmp_path)
    decision = PlannerDecision(
        action="tool_call",
        tool_call=ToolCall(
            name="create_draft_purchase_order",
            arguments={"sku": "GLOVE-M", "quantity": 2},
        ),
    )
    planner = FixedPlanner(decision)
    saver = sqlite_checkpointer(str(database.path))
    graph = build_graph(database, planner, PolicyRetriever(tmp_path / "no-index"), saver)

    first = invoke_chat(graph, "Create a draft order for 2 GLOVE-M because it is low stock", "thread-approval")
    pending = database.list_pending_approvals()
    assert len(pending) == 1
    order_id = pending[0]["order_id"]
    assert pending[0]["status"] == "pending"
    assert first.get("response", "").startswith("Draft ")
    assert graph.get_state({"configurable": {"thread_id": "thread-approval"}}).tasks
    trace = database.list_logs("thread-approval")
    assert any(entry["event"] == "tool_arguments_validated" for entry in trace)
    timed = [
        entry
        for entry in trace
        if entry["event"] in {"decision", "tool_call", "approval_requested"}
    ]
    assert all(entry["details"]["duration_ms"] >= 0 for entry in timed)

    saver.conn.close()
    saver = sqlite_checkpointer(str(database.path))
    graph = build_graph(database, planner, PolicyRetriever(tmp_path / "no-index"), saver)
    resumed = resume_approval(graph, "thread-approval", "YES")
    saver.conn.close()
    assert "approved" in resumed["response"]
    assert database.get_order(order_id)["status"] == "approved"
    assert database.get_pending_approval(order_id)["status"] == "approved"


def test_no_decision_rejects_order(tmp_path):
    database = seeded_database(tmp_path)
    decision = PlannerDecision(
        action="tool_call",
        tool_call=ToolCall(
            name="create_draft_purchase_order",
            arguments={"sku": "GLOVE-M", "quantity": 1},
        ),
    )
    graph = build_graph(database, FixedPlanner(decision), PolicyRetriever(tmp_path / "no-index"), MemorySaver())
    invoke_chat(graph, "Draft an order for GLOVE-M", "thread-no")
    order_id = database.list_pending_approvals()[0]["order_id"]
    resumed = resume_approval(graph, "thread-no", "NO")
    assert "rejected" in resumed["response"]
    assert database.get_order(order_id)["status"] == "rejected"


def test_model_errors_are_explicit():
    with pytest.raises(MissingAPIKeyError):
        create_llm(Settings(llm_provider="gemini", google_api_key=""))
    with pytest.raises(LLMUnavailableError, match="Unsupported"):
        create_llm(Settings(llm_provider="unknown"))


def test_unavailable_llm_invocation_is_typed(tmp_path):
    class UnavailablePlanner:
        def with_structured_output(self, _schema):
            return self

        def invoke(self, _messages):
            raise ConnectionError("provider unavailable")

    database = seeded_database(tmp_path)
    graph = build_graph(
        database,
        UnavailablePlanner(),
        PolicyRetriever(tmp_path / "no-index"),
        MemorySaver(),
    )
    with pytest.raises(LLMUnavailableError, match="Planner Agent"):
        invoke_chat(graph, "Check stock", "thread-llm-error")


def test_planner_schema_prevents_raw_sql_tool_payload():
    with pytest.raises(ValidationError):
        PlannerDecision.model_validate(
            {
                "action": "tool_call",
                "tool_call": {"name": "query_inventory", "arguments": {"sql": "SELECT 1"}},
            }
        )
    with pytest.raises(ValidationError, match="tool_call is required"):
        PlannerDecision.model_validate({"action": "tool_call"})


def test_planner_routes_again_after_inventory_tool_result(tmp_path):
    database = seeded_database(tmp_path)
    decisions = [
        PlannerDecision(
            action="tool_call",
            tool_call=ToolCall(name="query_inventory", arguments={"sku": "GLOVE-M"}),
        ),
        PlannerDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="create_draft_purchase_order",
                arguments={"sku": "GLOVE-M", "quantity": 5},
            ),
        ),
    ]

    class SequencePlanner(FixedPlanner):
        def __init__(self):
            self.decisions = iter(decisions)
            self.observed_messages = []

        def invoke(self, messages):
            self.observed_messages.append(messages)
            return next(self.decisions)

    planner = SequencePlanner()
    graph = build_graph(database, planner, PolicyRetriever(tmp_path / "no-index"), MemorySaver())
    invoke_chat(graph, "Check stock for GLOVE-M and draft an order if needed", "thread-routed")

    assert len(planner.observed_messages) == 2
    assert any(
        "GLOVE-M" in message.content
        for message in planner.observed_messages[1]
        if isinstance(message.content, str)
    )
    order_id = database.list_pending_approvals()[0]["order_id"]
    assert database.get_order(order_id)["items"][0]["quantity"] == 10
    assert [entry["event"] for entry in database.list_logs("thread-routed")].count("tool_call") == 2


def test_builtin_read_query_runs_real_agent_nodes_without_llm(tmp_path):
    database = seeded_database(tmp_path)

    class UnavailablePlanner:
        def with_structured_output(self, _schema):
            raise AssertionError("built-in read queries must not invoke the LLM")

    graph = build_graph(
        database,
        UnavailablePlanner(),
        PolicyRetriever(tmp_path / "no-index"),
        MemorySaver(),
    )
    result = invoke_chat(graph, "Show low-stock supplies", "thread-deterministic")
    logs = database.list_logs("thread-deterministic")
    assert result["planning_mode"] == "deterministic_read"
    assert result["response"].startswith("I found 1 low-stock supply item across 1 clinic.")
    assert not result["response"].startswith("[")
    assert [step["tool"] for step in result["execution_steps"] if step.get("agent") == "inventory"] == [
        "query_inventory"
    ]
    assert {entry["agent"] for entry in logs} >= {"planner", "inventory", "tool_guardrail"}
    assert any(entry["event"] == "decision" for entry in logs if entry["agent"] == "planner")
    assert any(entry["event"] == "tool_call" for entry in logs if entry["agent"] == "inventory")
    supplier_result = invoke_chat(
        graph,
        "Show supplier lead times and minimum order quantities",
        "thread-vendors",
    )
    assert supplier_result["response"] == (
        "I found 1 supplier. Lead times range from 4 to 4 days, and minimum order "
        "quantities range from 10 to 10 units."
    )
