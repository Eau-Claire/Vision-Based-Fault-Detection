import hashlib
from pathlib import Path
from typing import Dict

from edge.providers.roboflow.errors import ProviderCapabilityUnavailable, ProviderRetryableError
from edge.providers.roboflow.models import (
    BoundingBox,
    VisionDetection,
    WorkflowDescription,
    WorkflowRunRequest,
    WorkflowRunResult,
)


class FakeVisionWorkflowProvider:
    """Deterministic offline provider. This is not real model inference."""

    provider_name = "fake-vision-provider"

    def __init__(
        self,
        capability_available: bool = True,
        fail_times: int = 0,
        invalid_response: bool = False,
        empty_detections: bool = False,
    ):
        self.capability_available = capability_available
        self.fail_times = fail_times
        self.invalid_response = invalid_response
        self.empty_detections = empty_detections
        self.calls = 0

    def describe_workflow(self, workflow_ref: str) -> WorkflowDescription:
        return WorkflowDescription(
            workflow_ref=workflow_ref,
            input_names=["image"],
            parameter_schema={},
            output_names=["fake_predictions"],
            provider=self.provider_name,
            capability_available=self.capability_available,
            notes="Deterministic fake provider for offline harness tests.",
        )

    def run_workflow(self, request: WorkflowRunRequest) -> WorkflowRunResult:
        self.calls += 1
        if not self.capability_available:
            raise ProviderCapabilityUnavailable("Fake provider capability is unavailable")
        if self.calls <= self.fail_times:
            raise ProviderRetryableError("Fake provider retryable failure")
        if self.invalid_response:
            return WorkflowRunResult(
                workflow_ref=request.workflow_ref,
                provider=self.provider_name,
                detections=None,
                output_names=["fake_predictions"],
                is_fake=True,
                raw_summary={"invalid": True},
            )
        detections = [] if self.empty_detections else [self._detection_for(request.image_path)]
        return WorkflowRunResult(
            workflow_ref=request.workflow_ref,
            provider=self.provider_name,
            detections=detections,
            output_names=["fake_predictions"],
            is_fake=True,
            raw_summary={
                "fake": True,
                "imageName": request.image_path.name,
                "message": "Deterministic fake result; not model inference.",
            },
        )

    def _detection_for(self, image_path: Path) -> VisionDetection:
        digest = hashlib.sha256(image_path.read_bytes()).digest()
        x = (digest[0] % 50) / 100
        y = (digest[1] % 50) / 100
        width = 0.2 + (digest[2] % 20) / 100
        height = 0.2 + (digest[3] % 20) / 100
        confidence = 0.5 + (digest[4] % 45) / 100
        return VisionDetection(
            label="fake-object",
            confidence=round(min(confidence, 0.99), 2),
            bounding_box=BoundingBox(x=x, y=y, width=width, height=height),
            fake=True,
        )
