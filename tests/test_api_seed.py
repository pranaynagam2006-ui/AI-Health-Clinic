from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.db import Database
from app.graph import build_graph
from app.retrieval import PolicyRetriever
from app.schemas import PlannerDecision, ToolCall
from scripts.seed import DEFAULT_CSV, import_csv
from scripts.generate_sample_data import generate_csv


class FixedPlanner:
    def __init__(self, decision):
        self.decision = decision

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        return self.decision


def test_seed_import_loads_bundled_supplier_terms(tmp_path):
    database = Database(tmp_path / "seed.db")
    assert import_csv(database, DEFAULT_CSV) == 6
    records = database.query_inventory()
    assert len(records) == 6
    suppliers = {record["supplier_name"] for record in records}
    assert len(suppliers) == 3
    assert database.query_inventory(sku="SYR-10ML")[0]["minimum_order_quantity"] == 25
    assert len(database.list_clinics()) == 3
    assert {row["category"] for row in records} >= {"PPE", "Surgical", "Diagnostics"}
    assert {row["clinic_id"] for row in records} == {"CLINIC-001", "CLINIC-002", "CLINIC-003"}


def test_generated_dataset_is_configurable_and_importable(tmp_path):
    csv_path = tmp_path / "generated.csv"
    assert generate_csv(csv_path, count=37, seed=9) == 37
    database = Database(tmp_path / "generated.db")
    assert import_csv(database, csv_path) == 37
    records = database.query_inventory()
    assert len(records) == 37
    assert len({record["sku"] for record in records}) == 37
    assert len({record["clinic_id"] for record in records}) > 1
    assert len({record["supplier_name"] for record in records}) > 1
    assert len({record["lead_time_days"] for record in records}) > 1
    assert len({record["minimum_order_quantity"] for record in records}) > 1


def test_chat_surfaces_missing_gemini_key_as_typed_error(tmp_path):
    settings = Settings(
        llm_provider="gemini",
        google_api_key="",
        database_path=tmp_path / "api.db",
        policy_store_path=tmp_path / "index",
    )
    client = TestClient(create_app(settings=settings))
    with client:
        response = client.post("/api/query", json={"message": "Check inventory"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "missing_api_key"


def test_quota_error_schema_and_typed_classification():
    from app.errors import LLMQuotaExceededError

    error = LLMQuotaExceededError("Gemini quota exhausted")
    assert error.code == "llm_quota_exceeded"
    assert error.status_code == 429


def test_chat_exposes_pending_graph_interrupt(tmp_path):
    database = Database(tmp_path / "interrupt.db")
    database.initialize()
    database.upsert_inventory(
        sku="SKU-1",
        product_name="Test item",
        stock_level=1,
        daily_usage=1,
        lead_time_days=3,
        reorder_point=3,
        supplier_name="Supplier",
        supplier_email="orders@example.test",
        minimum_order_quantity=2,
        unit_cost_cents=100,
    )
    planner = FixedPlanner(
        PlannerDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="create_draft_purchase_order",
                arguments={"sku": "SKU-1", "quantity": 1},
            ),
        )
    )
    graph = build_graph(database, planner, PolicyRetriever(tmp_path / "no-index"))
    settings = Settings(database_path=tmp_path / "interrupt.db", policy_store_path=tmp_path / "no-index")
    client = TestClient(create_app(settings=settings, database=database, graph=graph))
    with client:
        response = client.post("/api/query", json={"message": "Draft an order for SKU-1"})
        approvals = client.get("/api/approvals")
    assert response.status_code == 200
    pending = response.json()["pending_approval"]
    assert pending["type"] == "purchase_order_approval"
    assert approvals.status_code == 200
    assert "Test item (SKU-1)" in approvals.json()[0]["items_summary"]
    approval = client.post(
        f"/api/approvals/{pending['order_id']}",
        json={"decision": "YES", "thread_id": response.json()["thread_id"]},
    )
    assert approval.status_code == 200
    assert approval.json()["order"]["status"] == "approved"


def test_fastapi_serves_dashboard_and_clinic_inventory_api(tmp_path):
    database = Database(tmp_path / "dashboard.db")
    assert import_csv(database, DEFAULT_CSV) == 6
    settings = Settings(
        llm_provider="gemini",
        google_api_key="",
        database_path=tmp_path / "dashboard.db",
        policy_store_path=tmp_path / "index",
    )
    client = TestClient(create_app(settings=settings, database=database))
    with client:
        page = client.get("/embed")
        low_stock = client.get("/api/inventory", params={"low_stock_only": "true"})
        clinics = client.get("/api/clinics")
        schema = client.get("/api/schema")
        docs = client.get("/openapi.json")
    assert page.status_code == 200
    assert "Embed this dashboard" in page.text
    assert "Validated tools" in page.text
    assert "Order writes via validated tools only" in page.text
    assert "x-frame-options" not in page.headers
    assert "Security guardrail simulator" in page.text
    assert "Relational schema inspector" in page.text
    assert "Planner Agent" in page.text
    assert "Inventory Agent" in page.text
    assert "Approval Guardrail Agent" in page.text
    assert "You're all caught up" in page.text
    assert "Approve order · YES" in page.text
    assert low_stock.status_code == 200
    assert all(row["stock_level"] <= row["reorder_point"] for row in low_stock.json())
    assert {clinic["clinic_id"] for clinic in clinics.json()} == {
        "CLINIC-001",
        "CLINIC-002",
        "CLINIC-003",
    }
    assert "/api/query" in docs.json()["paths"]
    assert "/api/approvals/{order_id}" in docs.json()["paths"]
    assert "/api/security/simulate" in docs.json()["paths"]
    assert "/api/schema" in docs.json()["paths"]
    table_names = {table["name"] for table in schema.json()}
    assert {"inventory", "clinics", "vendors", "purchase_orders", "agent_logs"} <= table_names
    purchase_order_columns = next(
        table["columns"] for table in schema.json() if table["name"] == "purchase_orders"
    )
    assert {"po_id", "clinic_id", "item_id", "vendor_id", "quantity", "status"} <= {
        column["name"] for column in purchase_order_columns
    }


def test_guardrail_simulator_endpoint_blocks_and_logs_attempt(tmp_path):
    database = Database(tmp_path / "guardrail.db")
    settings = Settings(
        llm_provider="gemini",
        google_api_key="",
        database_path=tmp_path / "guardrail.db",
        policy_store_path=tmp_path / "index",
    )
    client = TestClient(create_app(settings=settings, database=database))
    with client:
        response = client.post(
            "/api/security/simulate?thread_id=guardrail-test",
            json={"operation": "DELETE", "table": "inventory"},
        )
        logs = client.get("/api/logs", params={"thread_id": "guardrail-test"})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "destructive_operation_blocked"
    assert logs.json()[0]["event"] == "destructive_operation_blocked"
    assert logs.json()[0]["details"]["executed"] is False


def test_api_exposes_deterministic_low_stock_supplier_and_order_queries(tmp_path):
    database = Database(tmp_path / "deterministic.db")
    assert import_csv(database, DEFAULT_CSV) == 6
    settings = Settings(
        llm_provider="gemini",
        google_api_key="",
        database_path=tmp_path / "deterministic.db",
        policy_store_path=tmp_path / "index",
    )
    client = TestClient(create_app(settings=settings, database=database))
    with client:
        low_stock = client.get("/api/inventory", params={"low_stock_only": "true"})
        ppe = client.get("/api/inventory", params={"category": "PPE"})
        lakeshore = client.get("/api/inventory", params={"clinic_id": "CLINIC-003"})
        vendors = client.get("/api/vendors")
        orders = client.get("/api/orders", params={"status": "pending"})
    assert low_stock.status_code == 200
    assert all(row["stock_level"] <= row["reorder_point"] for row in low_stock.json())
    assert ppe.status_code == 200
    assert all(row["category"] == "PPE" for row in ppe.json())
    assert lakeshore.status_code == 200
    assert all(row["clinic_id"] == "CLINIC-003" for row in lakeshore.json())
    assert vendors.status_code == 200
    assert len(vendors.json()) == 3
    assert orders.status_code == 200
    assert isinstance(orders.json(), list)


def test_chat_runs_builtin_read_intent_without_llm_key_and_logs_agent_actions(tmp_path):
    database = Database(tmp_path / "deterministic-chat.db")
    assert import_csv(database, DEFAULT_CSV) == 6
    settings = Settings(
        llm_provider="gemini",
        google_api_key="",
        database_path=tmp_path / "deterministic-chat.db",
        policy_store_path=tmp_path / "index",
    )
    client = TestClient(create_app(settings=settings, database=database))
    with client:
        response = client.post("/api/query", json={"message": "Show low-stock supplies"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["execution_steps"][0]["planning_mode"] == "deterministic_read"
    assert payload["execution_steps"][-1]["tool"] == "query_inventory"
    assert payload["response"].startswith("I found ")
    assert "low-stock supply items" in payload["response"]
    agents = {entry["agent"] for entry in database.list_logs(payload["thread_id"])}
    assert {"planner", "inventory", "tool_guardrail"} <= agents
