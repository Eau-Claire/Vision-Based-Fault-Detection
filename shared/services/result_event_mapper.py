"""
Map normalized analysis results to RabbitMQ integration events.
"""

from datetime import datetime, timezone

from shared.schemas.analysis_result import AnalysisResult, AnalysisResultEvent


def map_result_to_event(result: AnalysisResult) -> AnalysisResultEvent:
    """Convert HTTP-callback-shaped result payload to RabbitMQ event payload."""
    processed_at = result.completed_at
    if isinstance(processed_at, str):
        processed_at = datetime.fromisoformat(
            processed_at.replace("Z", "+00:00")
        )
    if processed_at.tzinfo is None:
        processed_at = processed_at.replace(tzinfo=timezone.utc)

    return AnalysisResultEvent(
        correlation_id=result.correlation_id or result.request_id,
        analysis_id=result.request_id,
        inspection_id=result.mission_id or result.media_id or result.request_id,
        media_id=result.media_id,
        mission_id=result.mission_id,
        asset_id=result.asset_id,
        status=result.status,
        model_name=result.model_name,
        model_version=result.model_version,
        processing_time_ms=result.processing_time_ms,
        results=result.detections,
        video_metadata=result.video_metadata,
        raw_result=result.raw_result,
        error_code=result.error_code,
        error_message=result.error_message,
        processed_at=processed_at,
    )
