from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ToolResultStatus(str, Enum):
    SUCCESS = "success"
    RETRYABLE_FAILURE = "retryable_failure"
    PERMANENT_FAILURE = "permanent_failure"
    INVALID_INPUT = "invalid_input"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    DUPLICATE_PREVENTED = "duplicate_prevented"


@dataclass(frozen=True)
class ToolResult:
    status: ToolResultStatus
    output: Optional[Any] = None
    error_message: str = ""
    error_category: str = ""
    attempts: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == ToolResultStatus.SUCCESS
