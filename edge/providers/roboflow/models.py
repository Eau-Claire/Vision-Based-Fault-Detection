from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class VisionDetection:
    label: str
    confidence: float
    bounding_box: BoundingBox
    fake: bool = False


@dataclass(frozen=True)
class WorkflowDescription:
    workflow_ref: str
    input_names: List[str]
    parameter_schema: Dict[str, Any]
    output_names: List[str]
    provider: str
    capability_available: bool
    notes: str = ""


@dataclass(frozen=True)
class WorkflowRunRequest:
    workflow_ref: str
    image_path: Path
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowRunResult:
    workflow_ref: str
    provider: str
    detections: Optional[List[VisionDetection]]
    output_names: List[str]
    is_fake: bool = False
    raw_summary: Dict[str, Any] = field(default_factory=dict)
