from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ErrorCategory(str, Enum):
    INVALID_INPUT = "invalid_input"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    RETRYABLE = "retryable"
    PERMANENT = "permanent"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    VERIFICATION_FAILED = "verification_failed"
    DUPLICATE_PREVENTED = "duplicate_prevented"


@dataclass(frozen=True)
class ClassifiedError:
    category: ErrorCategory
    message: str
    retryable: bool = False
    provider: Optional[str] = None
    tool_name: Optional[str] = None


class HarnessError(Exception):
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.PERMANENT,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.classified = ClassifiedError(category, message, retryable=retryable)


class InvalidInputError(HarnessError):
    def __init__(self, message: str):
        super().__init__(message, ErrorCategory.INVALID_INPUT, retryable=False)


class CapabilityUnavailableError(HarnessError):
    def __init__(self, message: str):
        super().__init__(message, ErrorCategory.CAPABILITY_UNAVAILABLE, retryable=False)


class RetryableExecutionError(HarnessError):
    def __init__(self, message: str):
        super().__init__(message, ErrorCategory.RETRYABLE, retryable=True)


class VerificationFailedError(HarnessError):
    def __init__(self, message: str):
        super().__init__(message, ErrorCategory.VERIFICATION_FAILED, retryable=False)
