from typing import Dict

from edge.memory.models import CheckpointSnapshot


class InMemoryCheckpointStore:
    def __init__(self):
        self.snapshots: Dict[str, CheckpointSnapshot] = {}

    def save(self, snapshot: CheckpointSnapshot) -> str:
        self.snapshots[snapshot.run_id] = snapshot
        return f"memory://{snapshot.run_id}"

    def load_latest(self, run_id: str) -> CheckpointSnapshot:
        return self.snapshots[run_id]
