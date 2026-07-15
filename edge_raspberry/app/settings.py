"""
Edge-specific settings for the Raspberry Pi / YOLO11 runtime.

Extends BaseAppSettings with YOLO11 model paths, frame sampling config,
and edge-optimized inference parameters.
"""

from pydantic import Field
from shared.config.base_settings import BaseAppSettings


class EdgeSettings(BaseAppSettings):
    """Configuration for the edge_raspberry YOLO11 runtime."""

    # ── Device Identity ──
    device_profile: str = Field("edge", alias="DEVICE_PROFILE")
    runtime_name: str = Field("edge_raspberry", alias="RUNTIME_NAME")

    # ── YOLO11 Model ──
    yolo_model_path: str = Field(
        "models/yolo11/yolo11n.pt", alias="YOLO_MODEL_PATH"
    )
    yolo_image_size: int = Field(640, alias="YOLO_IMAGE_SIZE")
    yolo_conf_threshold: float = Field(0.25, alias="YOLO_CONF_THRESHOLD")
    yolo_iou_threshold: float = Field(0.45, alias="YOLO_IOU_THRESHOLD")
    yolo_device: str = Field("cpu", alias="YOLO_DEVICE")

    # ── Video Frame Sampling ──
    frame_sample_interval: int = Field(
        15, alias="FRAME_SAMPLE_INTERVAL",
        description="Process every Nth frame for video inference"
    )
    max_frames_per_video: int = Field(
        100, alias="MAX_FRAMES_PER_VIDEO",
        description="Maximum number of frames to process per video"
    )

    # ── Deduplication ──
    dedup_iou_threshold: float = Field(
        0.5, alias="DEDUP_IOU_THRESHOLD",
        description="IoU threshold for deduplicating nearby detections"
    )
    max_detections_per_class: int = Field(
        5, alias="MAX_DETECTIONS_PER_CLASS"
    )

    # ── Edge Server Port ──
    server_port: int = Field(8001, alias="SERVER_PORT")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


# Singleton
settings = EdgeSettings()
