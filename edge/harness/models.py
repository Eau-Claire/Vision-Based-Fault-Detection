from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class RunResultStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    ESCALATED = "escalated"


@dataclass(frozen=True)
class LocalImageTrigger:
    image_path: Path
    workflow_ref: str = "fake://evn-object-detection"
    goal: str = "Run vision workflow on local image"


@dataclass(frozen=True)
class ExecutionBudget:
    max_iterations: int = 1
    timeout_seconds: int = 30


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: RunResultStatus
    message: str
    checkpoint_path: Optional[Path] = None
    output: Optional[Any] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
