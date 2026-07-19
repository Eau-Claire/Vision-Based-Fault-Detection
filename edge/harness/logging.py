import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, TextIO

SECRET_MARKERS = ("api_key", "apikey", "token", "secret", "password", "passwd", "authorization")


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if any(marker in str(key).lower() for marker in SECRET_MARKERS):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = redact(item)
        return redacted
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    return value


@dataclass(frozen=True)
class StructuredEvent:
    event_type: str
    run_id: Optional[str] = None
    iteration_id: Optional[str] = None
    action_id: Optional[str] = None
    tool_name: Optional[str] = None
    provider: Optional[str] = None
    state_transition: Optional[str] = None
    duration_ms: Optional[int] = None
    retry_attempt: Optional[int] = None
    outcome: Optional[str] = None
    error_category: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )

    def to_dict(self) -> Dict[str, Any]:
        raw = asdict(self)
        return {key: redact(value) for key, value in raw.items() if value not in (None, {}, [])}


class StructuredEventLogger:
    def __init__(self, stream: Optional[TextIO] = None, emit: bool = False):
        self._stream = stream or sys.stdout
        self._emit = emit
        self.events: List[Dict[str, Any]] = []

    def emit(self, event_type: str, **kwargs: Any) -> None:
        event = StructuredEvent(event_type=event_type, **kwargs).to_dict()
        self.events.append(event)
        if self._emit:
            print(json.dumps(event, ensure_ascii=True), file=self._stream)

    def snapshot(self) -> List[Dict[str, Any]]:
        return list(self.events)
