"""
Edge Raspberry Pi — FastAPI application entry point.

Runs the YOLO11 inference service with:
- /health and /ready endpoints
- RabbitMQ consumer for edge analysis jobs
- REST API for ad-hoc analysis requests
"""

import os
import sys
import time
import threading

# Ensure project root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional

from edge_raspberry.app.settings import settings
from shared.utils.logging import setup_logging, get_logger, set_correlation_context
from shared.schemas.analysis_request import AnalysisRequest

# ── Logging ──
setup_logging(
    level=settings.log_level,
    log_format=settings.log_format,
    service_name="edge-raspberry",
)
logger = get_logger("main")

# ── Global state ──
detector = None
_ready = False


def _init_detector():
    """Load the YOLO11 model in a background thread."""
    global detector, _ready

    from edge_raspberry.app.detector import EdgeYoloDetector

    try:
        detector = EdgeYoloDetector(
            model_path=settings.yolo_model_path,
            image_size=settings.yolo_image_size,
            conf_threshold=settings.yolo_conf_threshold,
            iou_threshold=settings.yolo_iou_threshold,
            device=settings.yolo_device,
            frame_sample_interval=settings.frame_sample_interval,
            max_frames_per_video=settings.max_frames_per_video,
            dedup_iou_threshold=settings.dedup_iou_threshold,
            max_detections_per_class=settings.max_detections_per_class,
        )
        _ready = True
        logger.info(
            "YOLO11 detector initialized and ready",
            extra={"event": "detector_ready"},
        )
    except Exception as e:
        logger.error(
            f"Failed to initialize YOLO11 detector: {e}",
            exc_info=True,
            extra={"event": "detector_init_failed"},
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: init models and start consumer on startup."""
    logger.info("Edge Raspberry Pi service starting...")

    # Load model in background thread
    init_thread = threading.Thread(target=_init_detector, daemon=True)
    init_thread.start()

    # Wait for model to load (with timeout)
    init_thread.join(timeout=120)

    # Start RabbitMQ consumer if configured
    if settings.rabbitmq_host:
        from edge_raspberry.app.consumer import start_edge_consumer

        if detector is not None:
            start_edge_consumer(detector, settings)
        else:
            logger.warning(
                "Detector not loaded, RabbitMQ consumer not started",
                extra={"event": "consumer_skipped"},
            )
    else:
        logger.info(
            "RABBITMQ_HOST not configured, consumer disabled",
            extra={"event": "consumer_disabled"},
        )

    yield  # Application running

    logger.info("Edge Raspberry Pi service shutting down...")


# ── FastAPI App ──
app = FastAPI(
    title="Edge Raspberry Pi — YOLO11 AI Service",
    description="UAV Power-Line Inspection — Edge inference using YOLO11",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Health Endpoints ──
@app.get("/health")
def health():
    """Liveness check — returns 200 if the process is alive."""
    return {
        "status": "healthy",
        "runtime": "edge_raspberry",
        "model": "YOLO11",
    }


@app.get("/ready")
def ready():
    """Readiness check — returns 200 only if the model is loaded."""
    if not _ready or detector is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet",
        )
    return {
        "status": "ready",
        "runtime": "edge_raspberry",
        "model_name": detector.model_name,
        "model_version": detector.model_version,
    }


# ── Ad-hoc Analysis Endpoint ──
class AnalyzePayload(BaseModel):
    requestId: str
    mediaId: Optional[str] = None
    fileUrl: str
    mediaType: str = "Image"
    analysisType: str = "General"
    preferredModel: Optional[str] = None
    callbackUrl: Optional[str] = None
    correlationId: Optional[str] = None


@app.post("/api/analyze", status_code=202)
def analyze(payload: AnalyzePayload, background_tasks: BackgroundTasks):
    """Trigger ad-hoc analysis. Returns 202 Accepted immediately."""
    if not _ready or detector is None:
        raise HTTPException(503, detail="Model not loaded yet")

    background_tasks.add_task(_run_analysis, payload)

    return {
        "message": "Analysis started in background.",
        "requestId": payload.requestId,
        "runtime": "edge_raspberry",
    }


def _run_analysis(payload: AnalyzePayload):
    """Background task for ad-hoc analysis."""
    import cv2
    import numpy as np
    from shared.services.media_downloader import download_media, DownloadError
    from shared.services.callback_service import send_callback, CallbackError
    from shared.services.result_mapper import map_success_result, map_failure_result
    from edge_raspberry.app.video_processor import process_video

    set_correlation_context(
        correlation_id=payload.correlationId,
        request_id=payload.requestId,
    )

    start_time = time.monotonic()

    try:
        # Download
        file_bytes, ext = download_media(
            file_url=payload.fileUrl,
            base_url=settings.callback_base_url,
            timeout=settings.media_download_timeout,
            max_size_bytes=settings.media_max_size_bytes,
        )

        # Inference
        is_video = payload.mediaType.lower() == "video" or ext in (
            ".mp4", ".avi", ".mov", ".webm", ".mkv"
        )

        if is_video:
            detection_result = process_video(detector, file_bytes, ext)
        else:
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image")
            detection_result = detector.detect_image(image)

        processing_time_ms = int((time.monotonic() - start_time) * 1000)

        result = map_success_result(
            request_id=payload.requestId,
            media_id=payload.mediaId,
            detection_result=detection_result,
            model_name=detector.model_name,
            model_version=detector.model_version,
            processing_time_ms=processing_time_ms,
            device_profile="edge",
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        result = map_failure_result(
            request_id=payload.requestId,
            media_id=payload.mediaId,
            error_code="MODEL_INFERENCE_FAILED",
            error_message=str(e),
        )

    # Send callback
    callback_url = payload.callbackUrl or settings.callback_url
    try:
        send_callback(
            result=result,
            callback_url=callback_url,
            service_key=settings.ai_service_key,
            max_retries=settings.callback_max_retries,
        )
    except CallbackError:
        logger.error(f"Callback delivery failed for {payload.requestId}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.server_host,
        port=settings.server_port,
    )
