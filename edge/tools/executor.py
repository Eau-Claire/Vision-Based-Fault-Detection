import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Set

from edge.harness.errors import ErrorCategory, HarnessError
from edge.harness.events import EventTypes
from edge.harness.logging import StructuredEventLogger
from edge.harness.retry import RetryPolicy
from edge.tools.base import Idempotency
from edge.tools.policies import ToolExecutionPolicy
from edge.tools.registry import ToolRegistry
from edge.tools.results import ToolResult, ToolResultStatus


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        retry_policy: RetryPolicy,
        logger: StructuredEventLogger,
        policy: ToolExecutionPolicy | None = None,
    ):
        self.registry = registry
        self.retry_policy = retry_policy
        self.logger = logger
        self.policy = policy or ToolExecutionPolicy()
        self._completed_non_idempotent_actions: Set[str] = set()

    def execute(
        self,
        tool_name: str,
        action_id: str,
        input_data: Any,
        run_id: str,
        iteration_id: str,
    ) -> ToolResult:
        tool = self.registry.get(tool_name)
        if (
            self.policy.prevent_duplicate_non_idempotent
            and tool.metadata.idempotency == Idempotency.NON_IDEMPOTENT
            and action_id in self._completed_non_idempotent_actions
        ):
            return ToolResult(
                status=ToolResultStatus.DUPLICATE_PREVENTED,
                error_message=f"Duplicate non-idempotent action prevented: {action_id}",
                error_category=ErrorCategory.DUPLICATE_PREVENTED.value,
            )

        final_result = ToolResult(
            status=ToolResultStatus.PERMANENT_FAILURE,
            error_message="Tool did not execute",
            error_category=ErrorCategory.PERMANENT.value,
        )
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            delay = self.retry_policy.delay_for_attempt(attempt)
            if delay:
                time.sleep(delay)
            self.logger.emit(
                EventTypes.TOOL_STARTED,
                run_id=run_id,
                iteration_id=iteration_id,
                action_id=action_id,
                tool_name=tool_name,
                retry_attempt=attempt,
            )
            started = time.monotonic()
            result = self._execute_once(tool, input_data)
            duration_ms = int((time.monotonic() - started) * 1000)
            final_result = ToolResult(
                status=result.status,
                output=result.output,
                error_message=result.error_message,
                error_category=result.error_category,
                attempts=attempt,
                metadata=result.metadata,
            )
            event_type = EventTypes.TOOL_COMPLETED if final_result.ok else EventTypes.TOOL_FAILED
            self.logger.emit(
                event_type,
                run_id=run_id,
                iteration_id=iteration_id,
                action_id=action_id,
                tool_name=tool_name,
                retry_attempt=attempt,
                duration_ms=duration_ms,
                outcome=final_result.status.value,
                error_category=final_result.error_category or None,
            )
            if final_result.ok:
                if tool.metadata.idempotency == Idempotency.NON_IDEMPOTENT:
                    self._completed_non_idempotent_actions.add(action_id)
                return final_result
            if final_result.status != ToolResultStatus.RETRYABLE_FAILURE or not tool.metadata.retry_safe:
                return final_result

        return final_result

    def _execute_once(self, tool, input_data: Any) -> ToolResult:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(tool.execute, input_data)
        try:
            return future.result(timeout=tool.metadata.timeout_seconds)
        except FutureTimeout:
            future.cancel()
            return ToolResult(
                status=ToolResultStatus.TIMEOUT,
                error_message=f"Tool timed out after {tool.metadata.timeout_seconds}s",
                error_category=ErrorCategory.TIMEOUT.value,
            )
        except HarnessError as exc:
            return ToolResult(
                status=ToolResultStatus.RETRYABLE_FAILURE if exc.classified.retryable else ToolResultStatus.PERMANENT_FAILURE,
                error_message=exc.classified.message,
                error_category=exc.classified.category.value,
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
