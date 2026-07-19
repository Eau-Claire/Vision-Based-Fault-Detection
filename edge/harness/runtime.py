import argparse
from pathlib import Path

from edge.harness.logging import StructuredEventLogger
from edge.harness.models import LocalImageTrigger, RunResult
from edge.loop.controller import LocalVisionHarnessController
from edge.memory.file_store import FileCheckpointStore
from edge.providers.base import VisionWorkflowProvider


class HarnessRuntime:
    def __init__(
        self,
        repo_root: Path,
        checkpoint_dir: Path,
        provider: VisionWorkflowProvider | None = None,
        emit_logs: bool = False,
    ):
        self.logger = StructuredEventLogger(emit=emit_logs)
        self.controller = LocalVisionHarnessController(
            repo_root=repo_root,
            checkpoint_store=FileCheckpointStore(checkpoint_dir),
            provider=provider,
            logger=self.logger,
        )

    def start_run(self, trigger: LocalImageTrigger) -> RunResult:
        return self.controller.start(trigger)

    def resume_run(self, run_id: str) -> RunResult:
        return self.controller.resume(run_id)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the offline fake-provider harness slice")
    parser.add_argument("image_path")
    parser.add_argument("--checkpoint-dir", default="/tmp/vision-harness-checkpoints")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--emit-logs", action="store_true")
    args = parser.parse_args()

    runtime = HarnessRuntime(
        repo_root=Path(args.repo_root).resolve(),
        checkpoint_dir=Path(args.checkpoint_dir).resolve(),
        emit_logs=args.emit_logs,
    )
    result = runtime.start_run(LocalImageTrigger(image_path=Path(args.image_path).resolve()))
    print(f"run_id={result.run_id}")
    print(f"status={result.status.value}")
    print(f"message={result.message}")
    if result.checkpoint_path:
        print(f"checkpoint={result.checkpoint_path}")
    return 0 if result.status.value == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
