"""
Result mapper — converts internal DetectionResult to AnalysisResult.

Handles both success and failure cases, producing the normalized
callback payload expected by the ASP.NET Core backend.
"""

import time
from datetime import datetime, timezone
from typing import Optional

from shared.schemas.analysis_result import (
    AnalysisResult,
    AnalysisStatus,
    DetectionResult,
)
from shared.utils.logging import get_logger

logger = get_logger("result_mapper")


def map_success_result(
    request_id: str,
    media_id: Optional[str],
    detection_result: DetectionResult,
    model_name: str,
    model_version: str,
    processing_time_ms: int,
    device_profile: str = "",
) -> AnalysisResult:
    """Map a successful DetectionResult to an AnalysisResult callback payload.

    Args:
        request_id: The original analysis request ID.
        media_id: The media ID if available.
        detection_result: Internal detection result from the detector.
        model_name: Name of the model used.
        model_version: Version of the model used.
        processing_time_ms: Total processing time in milliseconds.
        device_profile: 'edge' or 'server' profile indicator.

    Returns:
        AnalysisResult ready for callback delivery.
    """
    return AnalysisResult(
        request_id=request_id,
        media_id=media_id,
        status=AnalysisStatus.COMPLETED,
        model_name=model_name,
        model_version=model_version,
        processing_time_ms=processing_time_ms,
        detections=detection_result.detections,
        raw_result={
            "imageWidth": detection_result.image_width,
            "imageHeight": detection_result.image_height,
            "frameCount": detection_result.frame_count,
            "deviceProfile": device_profile,
        },
        completed_at=datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    )


def map_failure_result(
    request_id: str,
    media_id: Optional[str],
    error_code: str,
    error_message: str,
) -> AnalysisResult:
    """Map an inference failure to an AnalysisResult callback payload.

    Args:
        request_id: The original analysis request ID.
        media_id: The media ID if available.
        error_code: Machine-readable error code.
        error_message: Human-readable error description.

    Returns:
        AnalysisResult with Failed status.
    """
    return AnalysisResult(
        request_id=request_id,
        media_id=media_id,
        status=AnalysisStatus.FAILED,
        error_code=error_code,
        error_message=error_message,
        completed_at=datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    )
