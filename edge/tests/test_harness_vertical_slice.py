import json
import tempfile
import unittest
from pathlib import Path

from edge.harness.logging import StructuredEventLogger, redact
from edge.harness.models import LocalImageTrigger, RunResultStatus
from edge.harness.retry import RetryPolicy
from edge.loop.controller import LocalVisionHarnessController
from edge.loop.state import LoopState, can_transition, transition, LoopRunState
from edge.memory.file_store import FileCheckpointStore
from edge.providers.roboflow.fake import FakeVisionWorkflowProvider
from edge.tools.base import Idempotency, SideEffectClass, ToolMetadata
from edge.tools.executor import ToolExecutor
from edge.tools.registry import ToolRegistry
from edge.tools.results import ToolResult, ToolResultStatus


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_IMAGE = REPO_ROOT / "dataset/valid/Pollution-Flashover/17069099h_jpg.rf.0oR7BZ630Hu9XS4IBja4.jpg"


class HarnessVerticalSliceTests(unittest.TestCase):
    def make_controller(self, tmpdir, provider=None, retry_policy=None):
        return LocalVisionHarnessController(
            repo_root=REPO_ROOT,
            checkpoint_store=FileCheckpointStore(Path(tmpdir)),
            provider=provider or FakeVisionWorkflowProvider(),
            logger=StructuredEventLogger(),
            retry_policy=retry_policy or RetryPolicy(max_attempts=3),
        )

    def test_successful_end_to_end_fake_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = self.make_controller(tmpdir)
            result = controller.start(LocalImageTrigger(image_path=FIXTURE_IMAGE))

            self.assertEqual(result.status, RunResultStatus.COMPLETED)
            self.assertTrue(result.checkpoint_path.exists())
            self.assertTrue(result.output.is_fake)
            self.assertGreaterEqual(len(result.output.detections), 1)
            event_types = [event["event_type"] for event in result.events]
            self.assertIn("run.started", event_types)
            self.assertIn("checkpoint.saved", event_types)
            self.assertIn("run.completed", event_types)

    def test_missing_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = self.make_controller(tmpdir)
            result = controller.start(LocalImageTrigger(image_path=REPO_ROOT / "missing.jpg"))
            self.assertEqual(result.status, RunResultStatus.FAILED)
            self.assertIn("does not exist", result.message)

    def test_unsupported_image_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_file = Path(tmpdir) / "input.txt"
            bad_file.write_text("not an image")
            controller = self.make_controller(tmpdir)
            result = controller.start(LocalImageTrigger(image_path=bad_file))
            self.assertEqual(result.status, RunResultStatus.FAILED)
            self.assertIn("Unsupported image type", result.message)

    def test_retryable_tool_failure_followed_by_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = FakeVisionWorkflowProvider(fail_times=1)
            controller = self.make_controller(
                tmpdir,
                provider=provider,
                retry_policy=RetryPolicy(max_attempts=2),
            )
            result = controller.start(LocalImageTrigger(image_path=FIXTURE_IMAGE))
            self.assertEqual(result.status, RunResultStatus.COMPLETED)
            self.assertEqual(provider.calls, 2)

    def test_retry_exhaustion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = FakeVisionWorkflowProvider(fail_times=3)
            controller = self.make_controller(
                tmpdir,
                provider=provider,
                retry_policy=RetryPolicy(max_attempts=2),
            )
            result = controller.start(LocalImageTrigger(image_path=FIXTURE_IMAGE))
            self.assertEqual(result.status, RunResultStatus.FAILED)
            self.assertIn("retryable failure", result.message)

    def test_capability_unavailable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = FakeVisionWorkflowProvider(capability_available=False)
            controller = self.make_controller(tmpdir, provider=provider)
            result = controller.start(LocalImageTrigger(image_path=FIXTURE_IMAGE))
            self.assertEqual(result.status, RunResultStatus.BLOCKED)
            self.assertIn("unavailable", result.message)

    def test_invalid_provider_response(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = FakeVisionWorkflowProvider(invalid_response=True)
            controller = self.make_controller(tmpdir, provider=provider)
            result = controller.start(LocalImageTrigger(image_path=FIXTURE_IMAGE))
            self.assertEqual(result.status, RunResultStatus.FAILED)
            self.assertIn("invalid workflow result", result.message)

    def test_verification_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = FakeVisionWorkflowProvider(empty_detections=True)
            controller = self.make_controller(tmpdir, provider=provider)
            result = controller.start(LocalImageTrigger(image_path=FIXTURE_IMAGE))
            self.assertEqual(result.status, RunResultStatus.FAILED)
            self.assertIn("contains no detections", result.message)

    def test_checkpoint_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = self.make_controller(tmpdir)
            result = controller.start(LocalImageTrigger(image_path=FIXTURE_IMAGE))
            data = json.loads(result.checkpoint_path.read_text())
            self.assertEqual(data["schema_version"], "harness-checkpoint-v1")
            self.assertEqual(data["state"], "COMPLETED")
            self.assertEqual(data["artifact_refs"][0]["type"], "image")

    def test_resume_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = self.make_controller(tmpdir)
            result = controller.start(LocalImageTrigger(image_path=FIXTURE_IMAGE))
            resumed = controller.resume(result.run_id)
            self.assertEqual(resumed.status, RunResultStatus.COMPLETED)
            self.assertEqual(resumed.message, "Run already completed from checkpoint")

    def test_secret_redaction(self):
        payload = {
            "ROBOFLOW_API_KEY": "rf_secret",
            "nested": {"password": "secret", "safe": "value"},
        }
        redacted = redact(payload)
        self.assertEqual(redacted["ROBOFLOW_API_KEY"], "[REDACTED]")
        self.assertEqual(redacted["nested"]["password"], "[REDACTED]")
        self.assertEqual(redacted["nested"]["safe"], "value")

    def test_state_transition_validity(self):
        state = LoopRunState(run_id="run-test", goal="test")
        self.assertTrue(can_transition(LoopState.CREATED, LoopState.CONTEXT_READY))
        transition(state, LoopState.CONTEXT_READY)
        with self.assertRaises(ValueError):
            transition(state, LoopState.COMPLETED)

    def test_prevention_of_duplicate_non_idempotent_actions(self):
        class NonIdempotentTool:
            metadata = ToolMetadata(
                name="side.effect",
                description="test side effect",
                input_schema={},
                output_schema={},
                timeout_seconds=5,
                retry_safe=False,
                side_effect=SideEffectClass.EXTERNAL_WRITE,
                idempotency=Idempotency.NON_IDEMPOTENT,
                required_capability="test.side_effect",
            )

            def __init__(self):
                self.calls = 0

            def execute(self, input_data):
                self.calls += 1
                return ToolResult(status=ToolResultStatus.SUCCESS, output={"ok": True})

        registry = ToolRegistry()
        tool = NonIdempotentTool()
        registry.register(tool)
        executor = ToolExecutor(registry, RetryPolicy(max_attempts=1), StructuredEventLogger())

        first = executor.execute("side.effect", "action-1", {}, "run-1", "iter-1")
        second = executor.execute("side.effect", "action-1", {}, "run-1", "iter-1")

        self.assertEqual(first.status, ToolResultStatus.SUCCESS)
        self.assertEqual(second.status, ToolResultStatus.DUPLICATE_PREVENTED)
        self.assertEqual(tool.calls, 1)


if __name__ == "__main__":
    unittest.main()
