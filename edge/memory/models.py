from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from edge.harness.logging import redact
from edge.loop.state import LoopState


SCHEMA_VERSION = "harness-checkpoint-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ActionRecord:
    action_id: str
    tool_name: str
    status: str
    summary: str


@dataclass(frozen=True)
class CheckpointSnapshot:
    run_id: str
    goal: str
    state: LoopState
    iteration: int
    completed_actions: List[ActionRecord]
    latest_verification: Optional[Dict[str, Any]]
    retry_counters: Dict[str, int]
    provider_capability_snapshot: Dict[str, Any]
    artifact_refs: List[Dict[str, str]]
    error_summary: List[str]
    created_at: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        return redact(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointSnapshot":
        actions = [ActionRecord(**item) for item in data.get("completed_actions", [])]
        return cls(
            run_id=data["run_id"],
            goal=data["goal"],
            state=LoopState(data["state"]),
            iteration=int(data["iteration"]),
            completed_actions=actions,
            latest_verification=data.get("latest_verification"),
            retry_counters=dict(data.get("retry_counters", {})),
            provider_capability_snapshot=dict(data.get("provider_capability_snapshot", {})),
            artifact_refs=list(data.get("artifact_refs", [])),
            error_summary=list(data.get("error_summary", [])),
            created_at=data.get("created_at", utc_now()),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )
