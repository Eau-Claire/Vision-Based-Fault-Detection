from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple


class LoopState(str, Enum):
    CREATED = "CREATED"
    CONTEXT_READY = "CONTEXT_READY"
    PLANNED = "PLANNED"
    ACTION_SELECTED = "ACTION_SELECTED"
    ACTION_RUNNING = "ACTION_RUNNING"
    VERIFYING = "VERIFYING"
    CHECKPOINTED = "CHECKPOINTED"
    COMPLETED = "COMPLETED"
    RETRY_PENDING = "RETRY_PENDING"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    ESCALATED = "ESCALATED"


ALLOWED_TRANSITIONS: Dict[LoopState, Tuple[LoopState, ...]] = {
    LoopState.CREATED: (LoopState.CONTEXT_READY, LoopState.FAILED),
    LoopState.CONTEXT_READY: (LoopState.PLANNED, LoopState.FAILED),
    LoopState.PLANNED: (LoopState.ACTION_SELECTED, LoopState.FAILED),
    LoopState.ACTION_SELECTED: (LoopState.ACTION_RUNNING, LoopState.BLOCKED, LoopState.FAILED),
    LoopState.ACTION_RUNNING: (
        LoopState.VERIFYING,
        LoopState.RETRY_PENDING,
        LoopState.BLOCKED,
        LoopState.FAILED,
    ),
    LoopState.RETRY_PENDING: (LoopState.ACTION_RUNNING, LoopState.FAILED),
    LoopState.VERIFYING: (LoopState.CHECKPOINTED, LoopState.FAILED, LoopState.ESCALATED),
    LoopState.CHECKPOINTED: (LoopState.COMPLETED, LoopState.FAILED),
    LoopState.COMPLETED: (),
    LoopState.BLOCKED: (),
    LoopState.FAILED: (),
    LoopState.ESCALATED: (),
}


@dataclass
class LoopRunState:
    run_id: str
    goal: str
    state: LoopState = LoopState.CREATED
    iteration: int = 0
    completed_actions: List[str] = field(default_factory=list)
    retry_counters: Dict[str, int] = field(default_factory=dict)
    error_summary: List[str] = field(default_factory=list)


def can_transition(current: LoopState, next_state: LoopState) -> bool:
    return next_state in ALLOWED_TRANSITIONS[current]


def transition(run_state: LoopRunState, next_state: LoopState) -> Tuple[LoopState, LoopState]:
    if not can_transition(run_state.state, next_state):
        raise ValueError(f"Invalid state transition: {run_state.state.value} -> {next_state.value}")
    previous = run_state.state
    run_state.state = next_state
    return previous, next_state
