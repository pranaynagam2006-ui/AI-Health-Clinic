import pytest
from pydantic import ValidationError

from app.db import Database
from app.errors import DestructiveOperationBlockedError, InvalidToolArgumentsError
from app.schemas import CreateDraftPurchaseOrderInput, InventoryQuery
from app.tools import ToolRegistry, calculate_reorder_point


def seeded_database(tmp_path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    database.upsert_inventory(
        sku="SKU-1",
        product_name="Test item",
        stock_level=3,
        daily_usage=2,
        lead_time_days=4,
        reorder_point=8,
        supplier_name="Supplier A",
        supplier_email="orders@example.test",
        minimum_order_quantity=10,
        unit_cost_cents=125,
    )
    return database


def test_tool_schemas_reject_unknown_and_invalid_arguments():
    with pytest.raises(ValidationError):
        InventoryQuery.model_validate({"sql": "SELECT * FROM inventory"})
    with pytest.raises(ValidationError):
        CreateDraftPurchaseOrderInput.model_validate({"sku": "SKU-1", "quantity": 0})
    with pytest.raises(ValidationError):
        InventoryQuery.model_validate({"clinic_id": "CLINIC-1", "unexpected": True})


def test_reorder_point_uses_usage_and_lead_time_and_checks_stock():
    registry = ToolRegistry(Database(":memory:"))
    result = registry._calculate_reorder_point(stock_level=7, daily_usage=2.2, lead_time_days=3)
    assert result == {"stock_level": 7, "reorder_point": 7, "needs_reorder": True}
    assert calculate_reorder_point(stock_level=100, daily_usage=0.2, lead_time_days=4) == 1


def test_purchase_order_respects_supplier_minimum_quantity(tmp_path):
    registry = ToolRegistry(seeded_database(tmp_path))
    result = registry.execute("create_draft_purchase_order", {"sku": "SKU-1", "quantity": 2})
    assert result["requested_quantity"] == 2
    assert result["quantity"] == 10
    assert result["total_cost_cents"] == 1250


def test_inventory_query_filters_low_stock_and_purchase_order_targets_clinic(tmp_path):
    database = seeded_database(tmp_path)
    registry = ToolRegistry(database)
    low_stock = registry.execute("query_inventory", {"low_stock_only": True})
    assert [row["sku"] for row in low_stock] == ["SKU-1"]
    assert [row["sku"] for row in registry.execute("query_inventory", {"search": "Test item"})] == ["SKU-1"]
    order = registry.execute(
        "create_draft_purchase_order",
        {"sku": "SKU-1", "quantity": 10, "clinic_id": "CLINIC-002"},
    )
    assert order["clinic_id"] == "CLINIC-002"
    assert order["clinic_name"] == "Downtown Family Clinic"
    stored = database.get_order(order["order_id"])
    assert stored["clinic_id"] == "CLINIC-002"


def test_tool_call_rejects_raw_sql_and_surfaces_typed_error(tmp_path):
    registry = ToolRegistry(seeded_database(tmp_path))
    with pytest.raises(InvalidToolArgumentsError, match="sql"):
        registry.execute(
            "query_inventory",
            {"sql": "DROP TABLE inventory"},
            thread_id="thread-invalid-tool",
        )
    log = registry.database.list_logs("thread-invalid-tool")[0]
    assert log["event"] == "tool_arguments_rejected"
    assert "DROP" not in str(log["details"])


def test_purchase_order_query_is_filtered_and_returns_vendor_and_clinic(tmp_path):
    database = seeded_database(tmp_path)
    order = database.create_draft_order("SKU-1", 10, clinic_id="CLINIC-002")
    database.record_pending_approval(order["order_id"], "thread-orders")
    registry = ToolRegistry(database)

    results = registry.execute(
        "query_purchase_orders",
        {"status": "pending", "vendor_name": "Supplier A", "clinic_id": "CLINIC-002"},
    )

    assert len(results) == 1
    assert results[0]["vendor_name"] == "Supplier A"
    assert results[0]["clinic_name"] == "Downtown Family Clinic"
    assert results[0]["items"][0]["sku"] == "SKU-1"


@pytest.mark.parametrize("operation", ["INSERT", "UPDATE", "DELETE"])
def test_destructive_operation_is_blocked_and_logged_at_tool_layer(tmp_path, operation):
    database = seeded_database(tmp_path)
    registry = ToolRegistry(database)

    with pytest.raises(DestructiveOperationBlockedError, match="blocked"):
        registry.reject_destructive_attempt(
            {"operation": operation, "table": "inventory"},
            "thread-guardrail-test",
        )

    logs = database.list_logs("thread-guardrail-test")
    assert logs[0]["event"] == "destructive_operation_blocked"
    assert logs[0]["details"]["executed"] is False
    assert logs[0]["details"]["table"] == "inventory"
    assert logs[0]["details"]["operation"] == operation
    assert len(database.query_inventory()) == 1


def test_unknown_sku_is_a_validation_error(tmp_path):
    registry = ToolRegistry(seeded_database(tmp_path))
    with pytest.raises(InvalidToolArgumentsError, match="Unknown SKU"):
        registry.execute("create_draft_purchase_order", {"sku": "NOPE", "quantity": 1})
