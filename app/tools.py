import math
from typing import Any, Callable

from pydantic import ValidationError

from app.db import Database
from app.errors import (
    DestructiveOperationBlockedError,
    InventoryAgentError,
    InvalidToolArgumentsError,
    UnknownToolError,
)
from app.schemas import (
    CreateDraftPurchaseOrderInput,
    DestructiveOperationAttempt,
    InventoryQuery,
    PurchaseOrderQuery,
    ReorderPointInput,
    VendorQuery,
)

TOOL_SCHEMAS = {
    "query_inventory": InventoryQuery,
    "query_purchase_orders": PurchaseOrderQuery,
    "query_vendors": VendorQuery,
    "calculate_reorder_point": ReorderPointInput,
    "create_draft_purchase_order": CreateDraftPurchaseOrderInput,
}


def calculate_reorder_point(stock_level: int, daily_usage: float, lead_time_days: int) -> int:
    """Return expected demand during supplier lead time."""
    if stock_level < 0 or daily_usage < 0 or lead_time_days <= 0:
        raise ValueError("Stock and daily usage must be non-negative and lead time must be positive")
    return math.ceil(daily_usage * lead_time_days)


class ToolRegistry:
    def __init__(self, database: Database):
        self.database = database
        self.handlers: dict[str, Callable[..., Any]] = {
            "query_inventory": self._query_inventory,
            "query_purchase_orders": self._query_purchase_orders,
            "query_vendors": self._query_vendors,
            "calculate_reorder_point": self._calculate_reorder_point,
            "create_draft_purchase_order": self._create_draft_purchase_order,
        }

    def execute(
        self,
        name: str,
        arguments: dict[str, object],
        thread_id: str | None = None,
    ) -> Any:
        schema = TOOL_SCHEMAS.get(name)
        handler = self.handlers.get(name)
        if schema is None or handler is None:
            if thread_id:
                self.database.log_event(
                    thread_id,
                    "tool_guardrail",
                    "unknown_tool_blocked",
                    {"tool": name, "argument_keys": sorted(arguments)},
                )
            raise UnknownToolError(f"Tool is not available: {name}")
        try:
            validated = schema.model_validate(arguments)
        except ValidationError as exc:
            if thread_id:
                self.database.log_event(
                    thread_id,
                    "tool_guardrail",
                    "tool_arguments_rejected",
                    {
                        "tool": name,
                        "schema": schema.__name__,
                        "argument_keys": sorted(arguments),
                        "validation_errors": [
                            {"location": error["loc"], "type": error["type"], "message": error["msg"]}
                            for error in exc.errors()
                        ],
                    },
                )
            raise InvalidToolArgumentsError(str(exc)) from exc
        if thread_id:
            self.database.log_event(
                thread_id,
                "tool_guardrail",
                "tool_arguments_validated",
                {"tool": name, "schema": schema.__name__, "valid": True},
            )
        try:
            return handler(**validated.model_dump())
        except InventoryAgentError as exc:
            if thread_id:
                self.database.log_event(
                    thread_id,
                    "tool_guardrail",
                    "tool_execution_failed",
                    {"tool": name, "error_code": exc.code, "message": exc.message},
                )
            raise

    def reject_destructive_attempt(
        self,
        arguments: dict[str, object],
        thread_id: str,
    ) -> None:
        try:
            attempt = DestructiveOperationAttempt.model_validate(arguments)
        except ValidationError as exc:
            self.database.log_event(
                thread_id,
                "tool_guardrail",
                "destructive_attempt_rejected",
                {
                    "argument_keys": sorted(arguments),
                    "validation_errors": [
                        {"location": error["loc"], "type": error["type"], "message": error["msg"]}
                        for error in exc.errors()
                    ],
                    "executed": False,
                },
            )
            raise InvalidToolArgumentsError(f"Invalid destructive-operation request: {exc}") from exc
        self.database.log_event(
            thread_id,
            "tool_guardrail",
            "destructive_operation_blocked",
            {
                "operation": attempt.operation,
                "table": attempt.table,
                "blocked": True,
                "executed": False,
            },
        )
        raise DestructiveOperationBlockedError(
            f"{attempt.operation} on {attempt.table} is blocked; only validated inventory and purchase-order tools may write."
        )

    def _query_inventory(
        self,
        sku: str | None = None,
        search: str | None = None,
        category: str | None = None,
        clinic_id: str | None = None,
        low_stock_only: bool = False,
    ) -> list[dict[str, Any]]:
        return self.database.query_inventory(
            sku=sku,
            search=search,
            category=category,
            clinic_id=clinic_id,
            low_stock_only=low_stock_only,
        )

    def _query_purchase_orders(
        self,
        status: str | None = None,
        vendor_name: str | None = None,
        clinic_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.database.query_orders(status=status, vendor_name=vendor_name, clinic_id=clinic_id)

    def _query_vendors(self, vendor_name: str | None = None) -> list[dict[str, Any]]:
        return self.database.list_vendors(vendor_name=vendor_name)

    @staticmethod
    def _calculate_reorder_point(stock_level: int, daily_usage: float, lead_time_days: int) -> dict[str, Any]:
        reorder_point = calculate_reorder_point(stock_level, daily_usage, lead_time_days)
        return {
            "stock_level": stock_level,
            "reorder_point": reorder_point,
            "needs_reorder": stock_level <= reorder_point,
        }

    def _create_draft_purchase_order(
        self,
        sku: str,
        quantity: int,
        clinic_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            return self.database.create_draft_order(sku, quantity, clinic_id=clinic_id)
        except ValueError as exc:
            raise InvalidToolArgumentsError(str(exc)) from exc
