from typing import Protocol

from edge.memory.models import CheckpointSnapshot


class CheckpointStore(Protocol):
    def save(self, snapshot: CheckpointSnapshot) -> str:
        ...

    def load_latest(self, run_id: str) -> CheckpointSnapshot:
        ...
