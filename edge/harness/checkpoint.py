from edge.memory.models import CheckpointSnapshot
from edge.memory.store import CheckpointStore


def save_checkpoint(store: CheckpointStore, snapshot: CheckpointSnapshot) -> str:
    return store.save(snapshot)
