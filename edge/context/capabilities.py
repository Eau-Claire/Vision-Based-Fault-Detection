from edge.context.models import CapabilitySnapshot
from edge.providers.base import VisionWorkflowProvider


def snapshot_provider_capabilities(provider: VisionWorkflowProvider, workflow_ref: str) -> CapabilitySnapshot:
    description = provider.describe_workflow(workflow_ref)
    return CapabilitySnapshot(
        provider=description.provider,
        capabilities={"vision.workflow.run": description.capability_available},
        notes=description.notes,
    )
