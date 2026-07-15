"""
Structured logging with correlationId and requestId propagation.

Outputs JSON-formatted log lines for production, or human-readable for dev.
"""

import logging
import json
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Optional

# Context variables for request-scoped correlation
_correlation_id: ContextVar[Optional[str]] = ContextVar(
    "correlation_id", default=None
)
_request_id: ContextVar[Optional[str]] = ContextVar(
    "request_id", default=None
)


def set_correlation_context(
    correlation_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> None:
    """Set correlation context for the current async/thread context."""
    if correlation_id is not None:
        _correlation_id.set(correlation_id)
    if request_id is not None:
        _request_id.set(request_id)


def clear_correlation_context() -> None:
    """Reset correlation context."""
    _correlation_id.set(None)
    _request_id.set(None)


class StructuredJsonFormatter(logging.Formatter):
    """Emit structured JSON log lines with correlation fields."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Inject correlation context
        cid = _correlation_id.get(None)
        rid = _request_id.get(None)
        if cid:
            log_entry["correlationId"] = cid
        if rid:
            log_entry["requestId"] = rid

        # Include exception info if present
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include extra fields
        for key in ("event", "model", "queue", "duration_ms", "status"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        return json.dumps(log_entry, ensure_ascii=False)


class ReadableFormatter(logging.Formatter):
    """Human-readable log format for local development."""

    def format(self, record: logging.LogRecord) -> str:
        cid = _correlation_id.get(None)
        rid = _request_id.get(None)

        parts = [
            f"[{record.levelname:<7}]",
            f"[{record.name}]",
        ]
        if rid:
            parts.append(f"[req:{rid[:8]}]")
        if cid:
            parts.append(f"[cor:{cid[:8]}]")

        parts.append(record.getMessage())

        msg = " ".join(parts)

        if record.exc_info and record.exc_info[1]:
            msg += "\n" + self.formatException(record.exc_info)

        return msg


def setup_logging(
    level: str = "INFO",
    log_format: str = "json",
    service_name: str = "ai-service",
) -> logging.Logger:
    """Configure the root logger with structured or readable formatting.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_format: 'json' for structured JSON, 'readable' for dev.
        service_name: Name prefix for the logger.

    Returns:
        Configured root logger.
    """
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    if log_format.lower() == "json":
        handler.setFormatter(StructuredJsonFormatter())
    else:
        handler.setFormatter(ReadableFormatter())

    logger.addHandler(handler)
    return logger


def get_logger(name: str, service_name: str = "ai-service") -> logging.Logger:
    """Get a child logger under the service namespace."""
    return logging.getLogger(f"{service_name}.{name}")
