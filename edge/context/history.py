from typing import List

from edge.loop.state import LoopRunState


def summarize_completed_actions(run_state: LoopRunState) -> List[str]:
    return [f"completed:{action_id}" for action_id in run_state.completed_actions]
