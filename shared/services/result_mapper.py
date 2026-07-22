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
    VideoMetadata,
)
from shared.utils.logging import get_logger

logger = get_logger("result_mapper")

CATEGORY_DISPLAY_NAMES = {
    "CI": "Cracked Insulator",
    "DS": "Damaged Splice",
    "VE": "Vegetation Encroachment",
    "CC": "Corroded Conductor",
    "BW": "Bird/Wildlife Damage",
    "SA": "Sagging Line",
    "ML": "Missing Label",
    "OT": "Other",
}


def map_success_result(
    request_id: str,
    media_id: Optional[str],
    detection_result: DetectionResult,
    model_name: str,
    model_version: str,
    processing_time_ms: int,
    device_profile: str = "",
    asset_id: Optional[str] = None,
    image_url: Optional[str] = None,
    tower_id: Optional[str] = None,
    gps: Optional[dict] = None,
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
        asset_id: Asset ID to copy onto each detection when available.
        image_url: Source or preview image URL for frontend display.
        tower_id: Tower ID to copy onto each detection when available.
        gps: GPS point to copy onto each detection when available.

    Returns:
        AnalysisResult ready for callback delivery.
    """
    detections = _enrich_detections(
        detection_result=detection_result,
        asset_id=asset_id,
        image_url=image_url,
        tower_id=tower_id,
        gps=gps,
    )
    video_metadata = VideoMetadata(
        duration=detection_result.duration,
        fps=detection_result.fps,
        width=detection_result.image_width,
        height=detection_result.image_height,
    )

    return AnalysisResult(
        request_id=request_id,
        media_id=media_id,
        status=AnalysisStatus.COMPLETED,
        model_name=model_name,
        model_version=model_version,
        processing_time_ms=processing_time_ms,
        detections=detections,
        video_metadata=video_metadata,
        raw_result={
            "imageWidth": detection_result.image_width,
            "imageHeight": detection_result.image_height,
            "frameCount": detection_result.frame_count,
            "fps": detection_result.fps,
            "duration": detection_result.duration,
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


def _enrich_detections(
    detection_result: DetectionResult,
    asset_id: Optional[str],
    image_url: Optional[str],
    tower_id: Optional[str],
    gps: Optional[dict],
):
    for detection in detection_result.detections:
        if not detection.class_name or detection.class_name == detection.category_code:
            detection.class_name = CATEGORY_DISPLAY_NAMES.get(
                detection.category_code, detection.category_code
            )
        if detection.timestamp is None and detection.timestamp_ms is not None:
            detection.timestamp = round(detection.timestamp_ms / 1000, 3)
        if detection.image_url is None:
            detection.image_url = image_url
        if detection.asset_id is None:
            detection.asset_id = asset_id
        if detection.tower_id is None:
            detection.tower_id = tower_id
        if detection.gps is None:
            detection.gps = gps
    return detection_result.detections
