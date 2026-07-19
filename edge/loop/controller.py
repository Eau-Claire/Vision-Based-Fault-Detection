from pathlib import Path
from typing import Optional
from uuid import uuid4

from edge.context.assembler import ContextAssembler
from edge.harness.events import EventTypes
from edge.harness.models import ExecutionBudget, LocalImageTrigger, RunResult, RunResultStatus
from edge.harness.logging import StructuredEventLogger
from edge.harness.retry import RetryPolicy
from edge.loop.planner import SimpleVisionPlanner
from edge.loop.policies import LocalImagePolicy
from edge.loop.state import LoopRunState, LoopState, transition
from edge.loop.verifier import VisionWorkflowVerifier
from edge.memory.models import ActionRecord, CheckpointSnapshot
from edge.memory.store import CheckpointStore
from edge.providers.base import VisionWorkflowProvider
from edge.providers.roboflow.fake import FakeVisionWorkflowProvider
from edge.tools.base_vision import VisionWorkflowTool
from edge.tools.executor import ToolExecutor
from edge.tools.registry import ToolRegistry
from edge.tools.results import ToolResultStatus


class LocalVisionHarnessController:
    def __init__(
        self,
        repo_root: Path,
        checkpoint_store: CheckpointStore,
        provider: Optional[VisionWorkflowProvider] = None,
        logger: Optional[StructuredEventLogger] = None,
        retry_policy: Optional[RetryPolicy] = None,
        budget: Optional[ExecutionBudget] = None,
    ):
        self.repo_root = repo_root
        self.checkpoint_store = checkpoint_store
        self.provider = provider or FakeVisionWorkflowProvider()
        self.logger = logger or StructuredEventLogger()
        self.retry_policy = retry_policy or RetryPolicy()
        self.budget = budget or ExecutionBudget()
        self.image_policy = LocalImagePolicy()
        self.planner = SimpleVisionPlanner()
        self.verifier = VisionWorkflowVerifier()
        self.context_assembler = ContextAssembler(repo_root, self.provider)

        registry = ToolRegistry()
        registry.register(VisionWorkflowTool(self.provider, timeout_seconds=self.budget.timeout_seconds))
        self.executor = ToolExecutor(registry, self.retry_policy, self.logger)

    def start(self, trigger: LocalImageTrigger) -> RunResult:
        image_path = trigger.image_path
        try:
            self.image_policy.validate(image_path)
        except Exception as exc:
            return RunResult(
                run_id="",
                status=RunResultStatus.FAILED,
                message=str(exc),
                events=self.logger.snapshot(),
            )

        run_id = f"run-{uuid4().hex}"
        iteration_id = f"{run_id}:iter-1"
        run_state = LoopRunState(run_id=run_id, goal=trigger.goal, iteration=1)
        self.logger.emit(EventTypes.RUN_STARTED, run_id=run_id, iteration_id=iteration_id)

        try:
            context = self.context_assembler.assemble(run_state, iteration_id, trigger.workflow_ref, self.budget)
            self._transition(run_state, LoopState.CONTEXT_READY, iteration_id)
            self.logger.emit(
                EventTypes.CONTEXT_ASSEMBLED,
                run_id=run_id,
                iteration_id=iteration_id,
                provider=context.capabilities.provider,
                details={"capabilities": context.capabilities.capabilities},
            )

            plan = self.planner.create_plan(run_id, image_path, trigger.workflow_ref)
            self._transition(run_state, LoopState.PLANNED, iteration_id)
            self.logger.emit(EventTypes.PLAN_CREATED, run_id=run_id, iteration_id=iteration_id)

            action = plan.actions[0]
            self._transition(run_state, LoopState.ACTION_SELECTED, iteration_id)
            self.logger.emit(
                EventTypes.ACTION_SELECTED,
                run_id=run_id,
                iteration_id=iteration_id,
                action_id=action.action_id,
                tool_name=action.tool_name,
            )

            self._transition(run_state, LoopState.ACTION_RUNNING, iteration_id)
            tool_result = self.executor.execute(
                action.tool_name,
                action.action_id,
                action.input_data,
                run_id,
                iteration_id,
            )
            if tool_result.status == ToolResultStatus.CAPABILITY_UNAVAILABLE:
                self._transition(run_state, LoopState.BLOCKED, iteration_id)
                return self._finish_without_checkpoint(run_state, RunResultStatus.BLOCKED, tool_result.error_message)
            if not tool_result.ok:
                self._transition(run_state, LoopState.FAILED, iteration_id)
                return self._finish_without_checkpoint(run_state, RunResultStatus.FAILED, tool_result.error_message)

            self._transition(run_state, LoopState.VERIFYING, iteration_id)
            verification = self.verifier.verify(tool_result)
            self.logger.emit(
                EventTypes.VERIFICATION_COMPLETED,
                run_id=run_id,
                iteration_id=iteration_id,
                action_id=action.action_id,
                outcome="passed" if verification.passed else "failed",
            )
            if not verification.passed:
                self._transition(run_state, LoopState.FAILED, iteration_id)
                return self._finish_without_checkpoint(run_state, RunResultStatus.FAILED, verification.message)

            run_state.completed_actions.append(action.action_id)
            self._transition(run_state, LoopState.CHECKPOINTED, iteration_id)
            snapshot = CheckpointSnapshot(
                run_id=run_id,
                goal=run_state.goal,
                state=run_state.state,
                iteration=run_state.iteration,
                completed_actions=[
                    ActionRecord(
                        action_id=action.action_id,
                        tool_name=action.tool_name,
                        status=tool_result.status.value,
                        summary="Vision workflow action completed",
                    )
                ],
                latest_verification={"passed": verification.passed, "message": verification.message},
                retry_counters={action.action_id: tool_result.attempts - 1},
                provider_capability_snapshot=context.capabilities.__dict__,
                artifact_refs=[{"type": "image", "path": str(image_path)}],
                error_summary=list(run_state.error_summary),
            )
            checkpoint_path = Path(self.checkpoint_store.save(snapshot))
            self.logger.emit(
                EventTypes.CHECKPOINT_SAVED,
                run_id=run_id,
                iteration_id=iteration_id,
                action_id=action.action_id,
                details={"checkpointPath": str(checkpoint_path)},
            )

            self._transition(run_state, LoopState.COMPLETED, iteration_id)
            completed = CheckpointSnapshot(
                run_id=snapshot.run_id,
                goal=snapshot.goal,
                state=run_state.state,
                iteration=snapshot.iteration,
                completed_actions=snapshot.completed_actions,
                latest_verification=snapshot.latest_verification,
                retry_counters=snapshot.retry_counters,
                provider_capability_snapshot=snapshot.provider_capability_snapshot,
                artifact_refs=snapshot.artifact_refs,
                error_summary=snapshot.error_summary,
            )
            checkpoint_path = Path(self.checkpoint_store.save(completed))
            self.logger.emit(EventTypes.RUN_COMPLETED, run_id=run_id, iteration_id=iteration_id)
            return RunResult(
                run_id=run_id,
                status=RunResultStatus.COMPLETED,
                message="Run completed",
                checkpoint_path=checkpoint_path,
                output=tool_result.output,
                events=self.logger.snapshot(),
            )
        except Exception as exc:
            run_state.error_summary.append(str(exc))
            if run_state.state not in (LoopState.FAILED, LoopState.BLOCKED, LoopState.COMPLETED):
                try:
                    self._transition(run_state, LoopState.FAILED, iteration_id)
                except ValueError:
                    run_state.state = LoopState.FAILED
            return self._finish_without_checkpoint(run_state, RunResultStatus.FAILED, str(exc))

    def resume(self, run_id: str) -> RunResult:
        snapshot = self.checkpoint_store.load_latest(run_id)
        if snapshot.state == LoopState.COMPLETED:
            return RunResult(
                run_id=run_id,
                status=RunResultStatus.COMPLETED,
                message="Run already completed from checkpoint",
                output=snapshot,
                events=self.logger.snapshot(),
            )
        return RunResult(
            run_id=run_id,
            status=RunResultStatus.BLOCKED,
            message=f"Resume requires a controller policy for state {snapshot.state.value}",
            output=snapshot,
            events=self.logger.snapshot(),
        )

    def _transition(self, run_state: LoopRunState, next_state: LoopState, iteration_id: str) -> None:
        previous, current = transition(run_state, next_state)
        self.logger.emit(
            EventTypes.STATE_TRANSITIONED,
            run_id=run_state.run_id,
            iteration_id=iteration_id,
            state_transition=f"{previous.value}->{current.value}",
        )

    def _finish_without_checkpoint(self, run_state: LoopRunState, status: RunResultStatus, message: str) -> RunResult:
        event_type = EventTypes.RUN_BLOCKED if status == RunResultStatus.BLOCKED else EventTypes.RUN_FAILED
        self.logger.emit(event_type, run_id=run_state.run_id, outcome=status.value)
        return RunResult(
            run_id=run_state.run_id,
            status=status,
            message=message,
            events=self.logger.snapshot(),
        )
