from pathlib import Path

from edge.context.capabilities import snapshot_provider_capabilities
from edge.context.history import summarize_completed_actions
from edge.context.models import ExecutionContext
from edge.context.repository_context import load_repository_instructions
from edge.harness.models import ExecutionBudget
from edge.loop.state import LoopRunState
from edge.providers.base import VisionWorkflowProvider


class ContextAssembler:
    def __init__(self, repo_root: Path, provider: VisionWorkflowProvider):
        self.repo_root = repo_root
        self.provider = provider

    def assemble(
        self,
        run_state: LoopRunState,
        iteration_id: str,
        workflow_ref: str,
        budget: ExecutionBudget,
    ) -> ExecutionContext:
        return ExecutionContext(
            run_id=run_state.run_id,
            iteration_id=iteration_id,
            goal=run_state.goal,
            state=run_state.state,
            previous_action_summaries=summarize_completed_actions(run_state),
            repository_instructions=load_repository_instructions(self.repo_root),
            capabilities=snapshot_provider_capabilities(self.provider, workflow_ref),
            remaining_iterations=max(budget.max_iterations - run_state.iteration, 0),
            error_history=list(run_state.error_summary),
        )
