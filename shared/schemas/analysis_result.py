"""
Analysis result DTOs — normalized output contract for both runtimes.

The callback payload sent to the ASP.NET Core backend uses these schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime, timezone
from enum import Enum


class AnalysisStatus(str, Enum):
    COMPLETED = "Completed"
    FAILED = "Failed"


class BoundingBox(BaseModel):
    """Normalized bounding box with values between 0 and 1."""

    x: float = Field(..., ge=0.0, le=1.0, description="Left edge (normalized)")
    y: float = Field(..., ge=0.0, le=1.0, description="Top edge (normalized)")
    width: float = Field(
        ..., ge=0.0, le=1.0, description="Width (normalized)"
    )
    height: float = Field(
        ..., ge=0.0, le=1.0, description="Height (normalized)"
    )


class Detection(BaseModel):
    """A single detection from model inference."""

    category_code: str = Field(..., alias="categoryCode")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: BoundingBox = Field(..., alias="boundingBox")
    timestamp_ms: Optional[int] = Field(None, alias="timestampMs")
    frame_index: Optional[int] = Field(None, alias="frameIndex")

    model_config = {"populate_by_name": True}


class AnalysisResult(BaseModel):
    """Callback payload sent to /api/internal/ai-analysis/results."""

    request_id: str = Field(..., alias="requestId")
    media_id: Optional[str] = Field(None, alias="mediaId")
    status: AnalysisStatus = Field(AnalysisStatus.COMPLETED)
    model_name: Optional[str] = Field(None, alias="modelName")
    model_version: Optional[str] = Field(None, alias="modelVersion")
    processing_time_ms: Optional[int] = Field(None, alias="processingTimeMs")
    detections: List[Detection] = Field(default_factory=list)
    raw_result: Optional[Any] = Field(None, alias="rawResult")
    error_code: Optional[str] = Field(None, alias="errorCode")
    error_message: Optional[str] = Field(None, alias="errorMessage")
    completed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        alias="completedAt",
    )

    model_config = {"populate_by_name": True}


class DetectionResult(BaseModel):
    """Internal detection result returned by Detector implementations.

    This is the common output schema that both EdgeYoloDetector and
    ServerRfDetrDetector produce before mapping to the callback payload.
    """

    detections: List[Detection] = Field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    frame_count: int = 0
