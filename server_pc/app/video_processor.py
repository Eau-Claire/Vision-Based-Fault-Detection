"""
Video processor for server_pc runtime.

Handles video file processing with denser frame sampling,
optimized for higher-accuracy detailed analysis.
"""

import os
from typing import Optional

from shared.schemas.analysis_result import DetectionResult
from shared.services.media_downloader import save_to_temp_file
from shared.utils.logging import get_logger

logger = get_logger("server_video_processor")


def process_video(
    detector,
    video_bytes: bytes,
    extension: str = ".mp4",
) -> DetectionResult:
    """Process a video file through the server detector.

    Saves the video bytes to a temp file, runs detector.detect_video(),
    then cleans up the temp file.

    Args:
        detector: ServerRfDetrDetector instance.
        video_bytes: Raw video file content.
        extension: Video file extension.

    Returns:
        DetectionResult from the detector.
    """
    temp_path: Optional[str] = None

    try:
        temp_path = save_to_temp_file(video_bytes, extension)
        logger.info(
            f"Saved video to temp file: {temp_path}",
            extra={"event": "video_temp_saved"},
        )

        result = detector.detect_video(temp_path)
        return result

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(
                "Cleaned up temp video file",
                extra={"event": "video_temp_cleaned"},
            )
