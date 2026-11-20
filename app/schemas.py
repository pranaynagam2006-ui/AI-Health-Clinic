from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class InventoryQuery(StrictModel):
    sku: str | None = Field(default=None, min_length=1, max_length=64)
    search: str | None = Field(default=None, min_length=1, max_length=120)
    category: str | None = Field(default=None, min_length=1, max_length=80)
    clinic_id: str | None = Field(default=None, min_length=1, max_length=64)
    low_stock_only: bool = False

    @field_validator("sku", "search", "category", "clinic_id")
    @classmethod
    def strip_filter(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None


class ReorderPointInput(StrictModel):
    stock_level: int = Field(ge=0)
    daily_usage: float = Field(ge=0)
    lead_time_days: int = Field(gt=0)


class PurchaseOrderQuery(StrictModel):
    status: Literal["draft", "pending", "approved", "rejected"] | None = None
    vendor_name: str | None = Field(default=None, min_length=1, max_length=160)
    clinic_id: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("vendor_name", "clinic_id")
    @classmethod
    def strip_order_filter(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None


class VendorQuery(StrictModel):
    vendor_name: str | None = Field(default=None, min_length=1, max_length=160)

    @field_validator("vendor_name")
    @classmethod
    def strip_vendor_name(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None


class CreateDraftPurchaseOrderInput(StrictModel):
    sku: str = Field(min_length=1, max_length=64)
    quantity: int = Field(gt=0, le=100_000)
    clinic_id: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("sku")
    @classmethod
    def strip_sku(cls, value: str) -> str:
        return value.strip()


class DestructiveOperationAttempt(StrictModel):
    operation: Literal["INSERT", "UPDATE", "DELETE"]
    table: Literal["inventory", "clinics", "vendors", "orders", "purchase_orders", "order_items"]


class ToolCall(StrictModel):
    name: Literal[
        "query_inventory",
        "query_purchase_orders",
        "query_vendors",
        "calculate_reorder_point",
        "create_draft_purchase_order",
    ]
    arguments: dict[str, object]

    @field_validator("arguments")
    @classmethod
    def validate_tool_arguments(cls, value: dict[str, object], info):
        schemas = {
            "query_inventory": InventoryQuery,
            "query_purchase_orders": PurchaseOrderQuery,
            "query_vendors": VendorQuery,
            "calculate_reorder_point": ReorderPointInput,
            "create_draft_purchase_order": CreateDraftPurchaseOrderInput,
        }
        name = info.data.get("name")
        schema = schemas.get(name)
        if schema is None:
            raise ValueError(f"Unknown tool: {name}")
        return schema.model_validate(value).model_dump()


class PlannerDecision(StrictModel):
    action: Literal["tool_call", "respond"]
    tool_call: ToolCall | None = None
    response: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def require_tool_for_action(self):
        if self.action == "tool_call" and self.tool_call is None:
            raise ValueError("tool_call is required when action is tool_call")
        if self.action == "respond" and self.tool_call is not None:
            raise ValueError("tool_call must be omitted when action is respond")
        return self


class ApprovalDecision(StrictModel):
    decision: Literal["YES", "NO"]

    @field_validator("decision", mode="before")
    @classmethod
    def normalize_decision(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value


class ChatRequest(StrictModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        return value.strip()


class ApprovalRequest(StrictModel):
    decision: Literal["YES", "NO"]
    thread_id: str = Field(min_length=1, max_length=128)

    @field_validator("decision", mode="before")
    @classmethod
    def normalize_decision(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value


class SeedProductRow(StrictModel):
    sku: str = Field(min_length=1, max_length=64)
    product_name: str = Field(min_length=1, max_length=160)
    stock_level: int = Field(ge=0)
    daily_usage: float = Field(ge=0)
    lead_time_days: int = Field(gt=0)
    unit_cost: float = Field(gt=0)
    supplier_name: str = Field(min_length=1, max_length=160)
    supplier_email: str = Field(min_length=1, max_length=254)
    minimum_order_quantity: int = Field(gt=0)
    reorder_point: int | None = Field(default=None, ge=0)
    category: str = Field(default="Other", min_length=1, max_length=80)
    clinic_id: str = Field(default="CLINIC-001", min_length=1, max_length=64)
    clinic_name: str = Field(default="Northside Community Clinic", min_length=1, max_length=160)
    city: str = Field(default="Seattle", min_length=1, max_length=120)
    region: str = Field(default="North", min_length=1, max_length=120)
    bed_count: int = Field(default=50, ge=0)
    vendor_rating: float = Field(default=4.5, ge=0, le=5)
    vendor_support_contact: str | None = Field(default=None, max_length=254)
