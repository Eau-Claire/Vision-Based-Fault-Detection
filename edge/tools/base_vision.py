from pathlib import Path
from typing import Any

from edge.harness.errors import ErrorCategory, InvalidInputError
from edge.providers.base import VisionWorkflowProvider
from edge.providers.roboflow.errors import ProviderCapabilityUnavailable, ProviderRetryableError
from edge.providers.roboflow.models import WorkflowRunRequest, WorkflowRunResult
from edge.tools.base import Idempotency, SideEffectClass, ToolMetadata
from edge.tools.results import ToolResult, ToolResultStatus


class VisionWorkflowTool:
    def __init__(self, provider: VisionWorkflowProvider, timeout_seconds: int = 10):
        self.provider = provider
        self.metadata = ToolMetadata(
            name="vision.workflow.run",
            description="Run a provider-neutral vision workflow on a local image path.",
            input_schema={"workflow_ref": "str", "image_path": "path"},
            output_schema={"workflow_ref": "str", "detections": "list"},
            timeout_seconds=timeout_seconds,
            retry_safe=True,
            side_effect=SideEffectClass.READ_ONLY,
            idempotency=Idempotency.IDEMPOTENT,
            required_capability="vision.workflow.run",
        )

    def execute(self, input_data: Any) -> ToolResult:
        if not isinstance(input_data, dict):
            return ToolResult(
                status=ToolResultStatus.INVALID_INPUT,
                error_message="Vision workflow input must be a dictionary",
                error_category=ErrorCategory.INVALID_INPUT.value,
            )
        try:
            request = WorkflowRunRequest(
                workflow_ref=str(input_data["workflow_ref"]),
                image_path=Path(input_data["image_path"]),
                parameters=dict(input_data.get("parameters", {})),
            )
        except KeyError as exc:
            return ToolResult(
                status=ToolResultStatus.INVALID_INPUT,
                error_message=f"Missing workflow input key: {exc}",
                error_category=ErrorCategory.INVALID_INPUT.value,
            )

        try:
            output = self.provider.run_workflow(request)
        except ProviderCapabilityUnavailable as exc:
            return ToolResult(
                status=ToolResultStatus.CAPABILITY_UNAVAILABLE,
                error_message=exc.classified.message,
                error_category=exc.classified.category.value,
                metadata={"provider": getattr(self.provider, "provider_name", "unknown")},
            )
        except ProviderRetryableError as exc:
            return ToolResult(
                status=ToolResultStatus.RETRYABLE_FAILURE,
                error_message=exc.classified.message,
                error_category=exc.classified.category.value,
                metadata={"provider": getattr(self.provider, "provider_name", "unknown")},
            )

        if not isinstance(output, WorkflowRunResult) or output.detections is None:
            return ToolResult(
                status=ToolResultStatus.PERMANENT_FAILURE,
                error_message="Provider returned an invalid workflow result",
                error_category=ErrorCategory.PERMANENT.value,
                metadata={"provider": getattr(self.provider, "provider_name", "unknown")},
            )

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=output,
            metadata={"provider": output.provider, "is_fake": output.is_fake},
        )
