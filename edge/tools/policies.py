from dataclasses import dataclass


@dataclass(frozen=True)
class ToolExecutionPolicy:
    prevent_duplicate_non_idempotent: bool = True
