import json
from pathlib import Path

from edge.memory.models import CheckpointSnapshot


class FileCheckpointStore:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, snapshot: CheckpointSnapshot) -> str:
        path = self._path_for(snapshot.run_id)
        path.write_text(json.dumps(snapshot.to_dict(), indent=2, ensure_ascii=True))
        return str(path)

    def load_latest(self, run_id: str) -> CheckpointSnapshot:
        path = self._path_for(run_id)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found for run: {run_id}")
        return CheckpointSnapshot.from_dict(json.loads(path.read_text()))

    def _path_for(self, run_id: str) -> Path:
        return self.checkpoint_dir / f"{run_id}.json"
