from pathlib import Path
from typing import List

from edge.context.models import RepositoryInstructionSnapshot


def load_repository_instructions(repo_root: Path) -> RepositoryInstructionSnapshot:
    files: List[str] = []
    for path in repo_root.rglob("AGENTS.md"):
        if ".git" not in path.parts and ".venv" not in path.parts:
            files.append(str(path.relative_to(repo_root)))
    note = "No repository-scoped AGENTS.md files found." if not files else "Repository instructions loaded."
    return RepositoryInstructionSnapshot(files=files, notes=note)
