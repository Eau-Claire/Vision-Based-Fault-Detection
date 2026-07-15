"""
Abstract Detector interface (Protocol).

Both EdgeYoloDetector and ServerRfDetrDetector must implement this protocol.
"""

from typing import Protocol, runtime_checkable
import numpy as np

from shared.schemas.analysis_result import DetectionResult


@runtime_checkable
class Detector(Protocol):
    """Common detector interface for both edge and server runtimes."""

    @property
    def model_name(self) -> str:
        """Human-readable model name (e.g. 'YOLO11n', 'RF-DETR-base')."""
        ...

    @property
    def model_version(self) -> str:
        """Model version string (e.g. '1.0.0')."""
        ...

    def detect_image(self, image: np.ndarray) -> DetectionResult:
        """Run inference on a single image (BGR numpy array).

        Args:
            image: OpenCV BGR image as numpy array.

        Returns:
            DetectionResult with normalized bounding boxes and classifications.
        """
        ...

    def detect_video(self, video_path: str) -> DetectionResult:
        """Run inference on a video file.

        Args:
            video_path: Absolute path to the video file on disk.

        Returns:
            DetectionResult with detections across sampled frames.
        """
        ...
