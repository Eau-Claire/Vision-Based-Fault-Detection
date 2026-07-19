"""
Server-specific settings for the PC / RF-DETR runtime.

Extends BaseAppSettings with RF-DETR model paths, GPU configuration,
and server-optimized inference parameters.
"""

from pydantic import Field
from shared.config.base_settings import BaseAppSettings


class ServerSettings(BaseAppSettings):
    """Configuration for the server_pc RF-DETR runtime."""

    # ── Device Identity ──
    device_profile: str = Field("server", alias="DEVICE_PROFILE")
    runtime_name: str = Field("server_pc", alias="RUNTIME_NAME")

    # ── Inference Backend ──
    inference_backend: str = Field(
        "harness",
        alias="SERVER_INFERENCE_BACKEND",
        description=(
            "harness for provider-independent runtime, roboflow for "
            "hosted Workflow, local for RF-DETR"
        ),
    )
    harness_checkpoint_dir: str = Field(
        "/tmp/vision-harness-checkpoints", alias="HARNESS_CHECKPOINT_DIR"
    )
    harness_workflow_ref: str = Field(
        "fake://evn-object-detection", alias="HARNESS_WORKFLOW_REF"
    )

    # ── RF-DETR Model ──
    rfdetr_model_path: str = Field(
        "models/rf_detr/rf_detr_base.pth", alias="RFDETR_MODEL_PATH"
    )
    rfdetr_config_path: str = Field(
        "", alias="RFDETR_CONFIG_PATH"
    )
    rfdetr_image_size: int = Field(560, alias="RFDETR_IMAGE_SIZE")
    rfdetr_conf_threshold: float = Field(0.3, alias="RFDETR_CONF_THRESHOLD")
    rfdetr_device: str = Field("auto", alias="RFDETR_DEVICE")

    # ── Video Frame Sampling (denser for server) ──
    frame_sample_interval: int = Field(
        5, alias="FRAME_SAMPLE_INTERVAL",
        description="Process every Nth frame (denser than edge)"
    )
    max_frames_per_video: int = Field(
        500, alias="MAX_FRAMES_PER_VIDEO",
        description="More frames for detailed analysis"
    )

    # ── Deduplication ──
    dedup_iou_threshold: float = Field(
        0.4, alias="DEDUP_IOU_THRESHOLD",
        description="IoU threshold for deduplicating nearby detections"
    )
    max_detections_per_class: int = Field(
        10, alias="MAX_DETECTIONS_PER_CLASS"
    )

    # ── Server Port ──
    server_port: int = Field(8002, alias="SERVER_PORT")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


# Singleton
settings = ServerSettings()
