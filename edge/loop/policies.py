from dataclasses import dataclass
from pathlib import Path

from edge.harness.errors import InvalidInputError


ALLOWED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})


@dataclass(frozen=True)
class LocalImagePolicy:
    allowed_suffixes: frozenset[str] = ALLOWED_IMAGE_SUFFIXES

    def validate(self, image_path: Path) -> None:
        if not image_path.exists():
            raise InvalidInputError(f"Image does not exist: {image_path}")
        if not image_path.is_file():
            raise InvalidInputError(f"Image path is not a file: {image_path}")
        if image_path.suffix.lower() not in self.allowed_suffixes:
            raise InvalidInputError(f"Unsupported image type: {image_path.suffix}")
