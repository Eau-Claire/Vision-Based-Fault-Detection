from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Protocol

from edge.tools.results import ToolResult


class SideEffectClass(str, Enum):
    READ_ONLY = "read_only"
    LOCAL_WRITE = "local_write"
    EXTERNAL_WRITE = "external_write"


class Idempotency(str, Enum):
    IDEMPOTENT = "idempotent"
    NON_IDEMPOTENT = "non_idempotent"


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    timeout_seconds: int
    retry_safe: bool
    side_effect: SideEffectClass
    idempotency: Idempotency
    required_capability: str
    redaction_rules: Dict[str, str] = field(default_factory=dict)


class Tool(Protocol):
    metadata: ToolMetadata

    def execute(self, input_data: Any) -> ToolResult:
        ...
