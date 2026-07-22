"""
Analysis result DTOs — normalized output contract for both runtimes.

The callback payload sent to the ASP.NET Core backend uses these schemas.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Any
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4


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

    id: str = Field(default_factory=lambda: str(uuid4()), alias="id")
    category_code: str = Field(..., alias="categoryCode")
    class_name: Optional[str] = Field(None, alias="class")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: BoundingBox = Field(..., alias="boundingBox")
    timestamp_ms: Optional[int] = Field(None, alias="timestampMs")
    timestamp: Optional[float] = None
    frame_index: Optional[int] = Field(None, alias="frameIndex")
    image_url: Optional[str] = Field(None, alias="imageUrl")
    crop_url: Optional[str] = Field(None, alias="cropUrl")
    gps: Optional[Any] = None
    tower_id: Optional[str] = Field(None, alias="towerId")
    asset_id: Optional[str] = Field(None, alias="assetId")

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def fill_frontend_fields(self):
        if self.class_name is None:
            self.class_name = self.category_code
        if self.timestamp is None and self.timestamp_ms is not None:
            self.timestamp = round(self.timestamp_ms / 1000, 3)
        return self


class VideoMetadata(BaseModel):
    """Video metadata used by frontend playback synchronization."""

    duration: Optional[float] = None
    fps: Optional[float] = None
    width: int = 0
    height: int = 0


class AnalysisResult(BaseModel):
    """Callback payload sent to /api/internal/ai-analysis/results."""

    request_id: str = Field(..., alias="requestId")
    media_id: Optional[str] = Field(None, alias="mediaId")
    status: AnalysisStatus = Field(AnalysisStatus.COMPLETED)
    model_name: Optional[str] = Field(None, alias="modelName")
    model_version: Optional[str] = Field(None, alias="modelVersion")
    processing_time_ms: Optional[int] = Field(None, alias="processingTimeMs")
    detections: List[Detection] = Field(default_factory=list)
    video_metadata: Optional[VideoMetadata] = Field(None, alias="videoMetadata")
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
    fps: Optional[float] = None
    duration: Optional[float] = None
