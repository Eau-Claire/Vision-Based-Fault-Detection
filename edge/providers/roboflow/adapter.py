from edge.harness.errors import CapabilityUnavailableError
from edge.providers.roboflow.models import WorkflowDescription, WorkflowRunRequest, WorkflowRunResult


class RoboflowWorkflowAdapter:
    """Adapter boundary for live Roboflow integration.

    TODO: When Roboflow MCP is connected, inspect the real workflow tools and
    map their actual request/response schemas here. Raw MCP payloads must not
    leak across this provider boundary.
    """

    provider_name = "roboflow"

    def describe_workflow(self, workflow_ref: str) -> WorkflowDescription:
        raise CapabilityUnavailableError(
            "Live Roboflow MCP schema is not wired into the generic adapter yet"
        )

    def run_workflow(self, request: WorkflowRunRequest) -> WorkflowRunResult:
        raise CapabilityUnavailableError(
            "Live Roboflow MCP execution is not wired into the generic adapter yet"
        )
