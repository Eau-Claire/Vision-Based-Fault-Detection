from dataclasses import dataclass, field
from typing import Dict, List

from edge.loop.state import LoopState


@dataclass(frozen=True)
class RepositoryInstructionSnapshot:
    files: List[str]
    notes: str


@dataclass(frozen=True)
class CapabilitySnapshot:
    provider: str
    capabilities: Dict[str, bool]
    notes: str = ""


@dataclass(frozen=True)
class ExecutionContext:
    run_id: str
    iteration_id: str
    goal: str
    state: LoopState
    previous_action_summaries: List[str]
    repository_instructions: RepositoryInstructionSnapshot
    capabilities: CapabilitySnapshot
    remaining_iterations: int
    error_history: List[str] = field(default_factory=list)
