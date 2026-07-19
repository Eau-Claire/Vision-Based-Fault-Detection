from dataclasses import dataclass
from pathlib import Path
from typing import List

from edge.loop.actions import LoopAction, build_vision_workflow_action


@dataclass(frozen=True)
class Plan:
    plan_id: str
    actions: List[LoopAction]


class SimpleVisionPlanner:
    def create_plan(self, run_id: str, image_path: Path, workflow_ref: str) -> Plan:
        return Plan(
            plan_id=f"{run_id}:plan:1",
            actions=[build_vision_workflow_action(run_id, image_path, workflow_ref)],
        )
