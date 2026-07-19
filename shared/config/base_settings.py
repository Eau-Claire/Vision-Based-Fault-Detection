"""
Base configuration shared between Edge and Server runtimes.

All settings are loaded from environment variables with sensible defaults.
Runtime-specific modules extend this with additional settings.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, model_validator
from typing import Optional


class BaseAppSettings(BaseSettings):
    """Common configuration for both edge_raspberry and server_pc runtimes."""

    # ── RabbitMQ ──
    rabbitmq_host: str = Field("localhost", alias="RABBITMQ_HOST")
    rabbitmq_port: int = Field(5672, alias="RABBITMQ_PORT")
    rabbitmq_user: str = Field("guest", alias="RABBITMQ_USER")
    rabbitmq_pass: str = Field("guest", alias="RABBITMQ_PASS")
    rabbitmq_heartbeat: int = Field(600, alias="RABBITMQ_HEARTBEAT")
    rabbitmq_prefetch_count: int = Field(1, alias="RABBITMQ_PREFETCH_COUNT")

    # ── RabbitMQ Queue Names ──
    rabbitmq_exchange: str = Field("ai.analysis", alias="RABBITMQ_EXCHANGE")
    edge_queue_name: str = Field(
        "ai.analysis.edge.requested", alias="EDGE_QUEUE_NAME"
    )
    server_queue_name: str = Field(
        "ai.analysis.server.requested", alias="SERVER_QUEUE_NAME"
    )
    dead_letter_exchange: str = Field(
        "ai.analysis.dlx", alias="DEAD_LETTER_EXCHANGE"
    )
    dead_letter_queue: str = Field(
        "ai.analysis.dead-letter", alias="DEAD_LETTER_QUEUE"
    )

    # ── Backend Callback ──
    callback_base_url: str = Field(
        "http://localhost:5000", alias="CALLBACK_BASE_URL"
    )
    callback_path: str = Field(
        "/api/internal/ai-analysis/results", alias="CALLBACK_PATH"
    )
    ai_service_key: str = Field("", alias="AI_SERVICE_KEY")

    # ── Callback Retry ──
    callback_max_retries: int = Field(3, alias="CALLBACK_MAX_RETRIES")
    callback_retry_base_delay: float = Field(
        1.0, alias="CALLBACK_RETRY_BASE_DELAY"
    )
    callback_retry_max_delay: float = Field(
        30.0, alias="CALLBACK_RETRY_MAX_DELAY"
    )
    callback_timeout: int = Field(15, alias="CALLBACK_TIMEOUT")

    # ── Media Download ──
    media_download_timeout: int = Field(60, alias="MEDIA_DOWNLOAD_TIMEOUT")
    media_max_size_bytes: int = Field(
        500 * 1024 * 1024, alias="MEDIA_MAX_SIZE_BYTES"  # 500 MB
    )

    # ── Security ──
    allow_private_ips: bool = Field(False, alias="ALLOW_PRIVATE_IPS")
    restrict_callback_to_base_url: bool = Field(True, alias="RESTRICT_CALLBACK_TO_BASE_URL")

    # ── Inference ──
    confidence_threshold: float = Field(
        0.25, alias="CONFIDENCE_THRESHOLD"
    )
    roboflow_api_key: str = Field("", alias="ROBOFLOW_API_KEY")
    roboflow_api_url: str = Field(
        "https://serverless.roboflow.com", alias="ROBOFLOW_API_URL"
    )
    roboflow_workspace_name: str = Field(
        "les-workspace-ijdwd", alias="ROBOFLOW_WORKSPACE_NAME"
    )
    roboflow_workflow_id: str = Field(
        "evn-object-detection-vevn-object-detection-cnyo0-1-rfdetr-small-t1-logic",
        alias="ROBOFLOW_WORKFLOW_ID",
    )
    roboflow_timeout: int = Field(30, alias="ROBOFLOW_TIMEOUT")
    roboflow_max_retries: int = Field(2, alias="ROBOFLOW_MAX_RETRIES")
    roboflow_retry_base_delay: float = Field(
        1.0, alias="ROBOFLOW_RETRY_BASE_DELAY"
    )

    # ── Server ──
    server_host: str = Field("0.0.0.0", alias="SERVER_HOST")
    server_port: int = Field(8000, alias="SERVER_PORT")

    # ── Logging ──
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    log_format: str = Field("json", alias="LOG_FORMAT")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }

    @model_validator(mode="after")
    def validate_security_settings(self):
        if not self.ai_service_key:
            raise ValueError("AI_SERVICE_KEY must be configured")
        if self.ai_service_key == "AI-Service-Secret-Token-Key-12345":
            raise ValueError("AI_SERVICE_KEY must not use the sample development secret")
        return self

    @property
    def callback_url(self) -> str:
        """Full callback URL built from base + path."""
        return f"{self.callback_base_url.rstrip('/')}{self.callback_path}"
