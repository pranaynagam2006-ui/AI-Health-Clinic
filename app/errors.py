class InventoryAgentError(Exception):
    """Base class for errors that are safe to return to API clients."""

    code = "inventory_agent_error"
    status_code = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class MissingAPIKeyError(InventoryAgentError):
    code = "missing_api_key"
    status_code = 503


class LLMUnavailableError(InventoryAgentError):
    code = "llm_unavailable"
    status_code = 503


class LLMQuotaExceededError(InventoryAgentError):
    code = "llm_quota_exceeded"
    status_code = 429


class InvalidToolArgumentsError(InventoryAgentError):
    code = "invalid_tool_arguments"
    status_code = 422


class UnknownToolError(InventoryAgentError):
    code = "unknown_tool"
    status_code = 422


class DestructiveOperationBlockedError(InventoryAgentError):
    code = "destructive_operation_blocked"
    status_code = 403


class ApprovalNotFoundError(InventoryAgentError):
    code = "approval_not_found"
    status_code = 404


class ApprovalConflictError(InventoryAgentError):
    code = "approval_conflict"
    status_code = 409


class DataImportError(InventoryAgentError):
    code = "data_import_error"
    status_code = 422
