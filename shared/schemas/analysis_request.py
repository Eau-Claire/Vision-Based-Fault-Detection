"""
Analysis request DTO — maps the RabbitMQ message payload.

All field names use camelCase to match the ASP.NET Core message contract.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Optional
from enum import Enum


class MediaType(str, Enum):
    IMAGE = "Image"
    VIDEO = "Video"


class AnalysisType(str, Enum):
    GENERAL = "General"
    DETAILED = "Detailed"
    URGENT = "Urgent"


class PreferredModel(str, Enum):
    YOLO11 = "YOLO11"
    EDGE = "EDGE"
    RF_DETR = "RF-DETR"
    SERVER = "SERVER"

    @classmethod
    def is_edge(cls, value: Optional[str]) -> bool:
        """Check if the preferred model targets the edge runtime."""
        if not value:
            return False
        return value.upper() in (cls.YOLO11.value.upper(), cls.EDGE.value.upper())

    @classmethod
    def is_server(cls, value: Optional[str]) -> bool:
        """Check if the preferred model targets the server runtime."""
        if not value:
            return False
        return value.upper() in (cls.RF_DETR.value.upper(), cls.SERVER.value.upper())


class AnalysisRequest(BaseModel):
    """Incoming analysis job from RabbitMQ queue."""

    request_id: str = Field(..., alias="requestId")
    media_id: Optional[str] = Field(None, alias="mediaId")
    mission_id: Optional[str] = Field(None, alias="missionId")
    asset_id: Optional[str] = Field(None, alias="assetId")
    file_url: str = Field(..., alias="fileUrl")
    media_type: MediaType = Field(MediaType.IMAGE, alias="mediaType")
    analysis_type: AnalysisType = Field(
        AnalysisType.GENERAL, alias="analysisType"
    )
    preferred_model: Optional[str] = Field(None, alias="preferredModel")
    callback_url: Optional[str] = Field(None, alias="callbackUrl")
    correlation_id: Optional[str] = Field(None, alias="correlationId")

    model_config = {"populate_by_name": True}

    @model_validator(mode="before")
    @classmethod
    def normalize_keys(cls, data):
        """Map PascalCase and camelCase keys case-insensitively to Python field names."""
        if isinstance(data, dict):
            mapping = {
                "requestid": "request_id",
                "mediaid": "media_id",
                "missionid": "mission_id",
                "assetid": "asset_id",
                "fileurl": "file_url",
                "mediatype": "media_type",
                "analysistype": "analysis_type",
                "preferredmodel": "preferred_model",
                "callbackurl": "callback_url",
                "correlationid": "correlation_id",
            }
            normalized = {}
            for k, v in data.items():
                k_lower = k.lower()
                if k_lower in mapping:
                    normalized[mapping[k_lower]] = v
                else:
                    normalized[k] = v
            return normalized
        return data
