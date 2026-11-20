import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from app.config import Settings
from app.db import Database
from app.errors import (
    ApprovalConflictError,
    ApprovalNotFoundError,
    InventoryAgentError,
    LLMUnavailableError,
)
from app.graph import build_graph, invoke_chat, resume_approval, sqlite_checkpointer
from app.llm import create_llm
from app.retrieval import PolicyRetriever
from app.schemas import ApprovalRequest, ChatRequest, DestructiveOperationAttempt
from app.tools import ToolRegistry

_DASHBOARD_FILE = Path(__file__).parent / "static" / "dashboard.html"


def create_app(
    settings: Settings | None = None,
    database: Database | None = None,
    llm: Any | None = None,
    graph: Any | None = None,
) -> FastAPI:
    settings = settings or Settings()
    database = database or Database(settings.database_path)
    resources: dict[str, Any] = {"graph": graph}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        database.initialize()
        if resources["graph"] is None:
            model = llm
            llm_error = None
            try:
                if model is None:
                    model = create_llm(settings)
            except InventoryAgentError as exc:
                llm_error = exc
            saver = sqlite_checkpointer(str(settings.database_path))
            resources["saver"] = saver
            resources["graph"] = build_graph(
                database,
                model,
                PolicyRetriever(settings.policy_store_path),
                checkpointer=saver,
                approval_threshold=settings.approval_threshold,
                llm_error=llm_error,
            )
        yield
        saver = resources.get("saver")
        if saver is not None:
            saver.conn.close()

    api = FastAPI(
        title="AI DataAssist — HealthPulse Clinical Operations",
        version="1.0.0",
        description="Clinical supply inventory, vendor purchase orders, and human approval workflow.",
        lifespan=lifespan,
    )

    @api.exception_handler(InventoryAgentError)
    async def inventory_error_handler(_: Request, exc: InventoryAgentError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @api.get("/", include_in_schema=False)
    @api.get("/dashboard", include_in_schema=False)
    @api.get("/embed", include_in_schema=False)
    def dashboard():
        return FileResponse(_DASHBOARD_FILE, media_type="text/html")

    @api.get("/health")
    def health():
        return {"status": "ok"}

    @api.get("/api/clinics")
    def clinics():
        return database.list_clinics()

    @api.get("/api/schema")
    def schema():
        return database.inspect_schema()

    @api.get("/api/vendors")
    def vendors():
        return database.list_vendors()

    @api.get("/api/orders")
    def orders(
        status: str | None = None,
        vendor_name: str | None = None,
        clinic_id: str | None = None,
    ):
        return database.query_orders(
            status=status,
            vendor_name=vendor_name,
            clinic_id=clinic_id,
        )

    @api.get("/api/inventory")
    @api.get("/inventory", include_in_schema=False)
    def inventory(
        sku: str | None = None,
        search: str | None = None,
        category: str | None = None,
        clinic_id: str | None = None,
        low_stock_only: bool = False,
    ):
        return database.query_inventory(
            sku=sku,
            search=search,
            category=category,
            clinic_id=clinic_id,
            low_stock_only=low_stock_only,
        )

    @api.post("/api/query")
    @api.post("/chat")
    def chat(body: ChatRequest):
        graph_app = resources.get("graph")
        if graph_app is None:
            raise LLMUnavailableError(
                resources.get(
                    "initialization_error",
                    "The configured LLM backend could not be initialized. Check its configuration and availability.",
                )
            )
        thread_id = body.thread_id or str(uuid.uuid4())
        state = invoke_chat(graph_app, body.message, thread_id)
        snapshot = graph_app.get_state({"configurable": {"thread_id": thread_id}})
        interrupts = [
            interrupt.value
            for task in snapshot.tasks
            for interrupt in task.interrupts
        ]
        return {
            "thread_id": thread_id,
            "response": state.get("response", ""),
            "pending_approval": interrupts[0] if interrupts else None,
            "execution_steps": state.get("execution_steps", []),
        }

    @api.get("/api/approvals")
    @api.get("/approvals", include_in_schema=False)
    def pending_approvals():
        return database.list_pending_approvals()

    @api.post("/api/approvals/{order_id}")
    @api.post("/approvals/{order_id}", include_in_schema=False)
    def decide_approval(order_id: str, body: ApprovalRequest):
        pending = database.get_pending_approval(order_id)
        if pending is None:
            raise ApprovalNotFoundError(f"No approval exists for order {order_id}.")
        if pending["status"] != "pending":
            raise ApprovalConflictError(f"Order {order_id} has already been resolved.")
        if pending["thread_id"] != body.thread_id:
            raise ApprovalConflictError("thread_id does not match the pending order.")
        graph_app = resources.get("graph")
        if graph_app is None:
            raise LLMUnavailableError("The workflow graph is unavailable; restart the API and try again.")
        state = resume_approval(graph_app, body.thread_id, body.decision)
        return {"order": database.get_order(order_id), "response": state.get("response", "")}

    @api.get("/api/orders/{order_id}")
    @api.get("/orders/{order_id}", include_in_schema=False)
    def order_detail(order_id: str):
        result = database.get_order(order_id)
        if result is None:
            raise ApprovalNotFoundError(f"Order {order_id} was not found.")
        return result

    @api.get("/api/logs")
    @api.get("/logs", include_in_schema=False)
    def logs(thread_id: str | None = None):
        return database.list_logs(thread_id)

    @api.post(
        "/api/security/simulate",
        responses={403: {"description": "Destructive operation blocked by the tool guardrail."}},
    )
    def simulate_destructive_operation(
        body: DestructiveOperationAttempt,
        thread_id: str | None = None,
    ):
        attempt_id = thread_id or f"guardrail-sim-{uuid.uuid4().hex}"
        ToolRegistry(database).reject_destructive_attempt(body.model_dump(), attempt_id)

    return api


app = create_app()
