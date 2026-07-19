from dataclasses import dataclass

from edge.providers.roboflow.models import WorkflowRunResult
from edge.tools.results import ToolResult, ToolResultStatus


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    message: str


class VisionWorkflowVerifier:
    def verify(self, result: ToolResult) -> VerificationResult:
        if result.status != ToolResultStatus.SUCCESS:
            return VerificationResult(False, f"Tool did not succeed: {result.status.value}")
        if not isinstance(result.output, WorkflowRunResult):
            return VerificationResult(False, "Tool output is not a WorkflowRunResult")
        if result.output.detections is None:
            return VerificationResult(False, "Provider result has no detections field")
        if not result.output.detections:
            return VerificationResult(False, "Provider result contains no detections")
        if result.output.is_fake and not result.output.raw_summary.get("fake"):
            return VerificationResult(False, "Fake provider result is not clearly marked fake")
        return VerificationResult(True, "Workflow result verified")
