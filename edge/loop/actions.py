from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict


class ActionType(str, Enum):
    RUN_VISION_WORKFLOW = "run_vision_workflow"


@dataclass(frozen=True)
class LoopAction:
    action_id: str
    action_type: ActionType
    tool_name: str
    input_data: Dict[str, Any] = field(default_factory=dict)


def build_vision_workflow_action(run_id: str, image_path: Path, workflow_ref: str) -> LoopAction:
    return LoopAction(
        action_id=f"{run_id}:vision-workflow:1",
        action_type=ActionType.RUN_VISION_WORKFLOW,
        tool_name="vision.workflow.run",
        input_data={"workflow_ref": workflow_ref, "image_path": str(image_path)},
    )
