from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.0
    max_delay_seconds: float = 5.0

    def delay_for_attempt(self, attempt: int) -> float:
        if attempt <= 1:
            return 0.0
        return min(self.base_delay_seconds * (2 ** (attempt - 2)), self.max_delay_seconds)
