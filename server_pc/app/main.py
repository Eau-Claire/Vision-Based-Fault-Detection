"""
Server PC — FastAPI application entry point.

Runs the RF-DETR inference service with:
- /health and /ready endpoints
- RabbitMQ consumer for server analysis jobs
- REST API for ad-hoc analysis requests
"""

import os
import sys
import time
import threading

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional

from server_pc.app.settings import settings
from shared.utils.logging import setup_logging, get_logger, set_correlation_context

setup_logging(
    level=settings.log_level,
    log_format=settings.log_format,
    service_name="server-pc",
)
logger = get_logger("main")

detector = None
_ready = False


def _init_detector():
    global detector, _ready
    from server_pc.app.detector import ServerRfDetrDetector
    try:
        detector = ServerRfDetrDetector(
            model_path=settings.rfdetr_model_path,
            config_path=settings.rfdetr_config_path,
            image_size=settings.rfdetr_image_size,
            conf_threshold=settings.rfdetr_conf_threshold,
            device=settings.rfdetr_device,
            frame_sample_interval=settings.frame_sample_interval,
            max_frames_per_video=settings.max_frames_per_video,
            dedup_iou_threshold=settings.dedup_iou_threshold,
            max_detections_per_class=settings.max_detections_per_class,
        )
        _ready = True
        logger.info("RF-DETR detector initialized", extra={"event": "detector_ready"})

        # Start consumer now that detector is ready!
        if settings.rabbitmq_host:
            from server_pc.app.consumer import start_server_consumer
            start_server_consumer(detector, settings)
    except Exception as e:
        logger.error(f"Failed to init RF-DETR: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server PC service starting...")
    t = threading.Thread(target=_init_detector, daemon=True)
    t.start()
    yield
    logger.info("Server PC service shutting down...")


app = FastAPI(
    title="Server PC — RF-DETR AI Service",
    description="UAV Power-Line Inspection — Server inference using RF-DETR",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "healthy", "runtime": "server_pc", "model": "RF-DETR"}


@app.get("/ready")
def ready():
    if not _ready or detector is None:
        raise HTTPException(503, detail="Model not loaded yet")
    return {
        "status": "ready", "runtime": "server_pc",
        "model_name": detector.model_name, "model_version": detector.model_version,
    }


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
    if not _ready or detector is None:
        raise HTTPException(503, detail="Model not loaded yet")
    background_tasks.add_task(_run_analysis, payload)
    return {"message": "Analysis started.", "requestId": payload.requestId, "runtime": "server_pc"}


def _run_analysis(payload: AnalyzePayload):
    import cv2, numpy as np
    from shared.services.media_downloader import download_media
    from shared.services.callback_service import send_callback, CallbackError
    from shared.services.result_mapper import map_success_result, map_failure_result
    from server_pc.app.video_processor import process_video

    set_correlation_context(correlation_id=payload.correlationId, request_id=payload.requestId)
    start = time.monotonic()
    try:
        file_bytes, ext = download_media(
            file_url=payload.fileUrl, base_url=settings.callback_base_url,
            timeout=settings.media_download_timeout, max_size_bytes=settings.media_max_size_bytes,
        )
        is_video = payload.mediaType.lower() == "video" or ext in (".mp4", ".avi", ".mov", ".webm")
        if is_video:
            dr = process_video(detector, file_bytes, ext)
        else:
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image")
            dr = detector.detect_image(image)

        ms = int((time.monotonic() - start) * 1000)
        result = map_success_result(
            payload.requestId, payload.mediaId, dr,
            detector.model_name, detector.model_version, ms, "server",
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        result = map_failure_result(payload.requestId, payload.mediaId, "MODEL_INFERENCE_FAILED", str(e))

    try:
        send_callback(result=result, callback_url=payload.callbackUrl or settings.callback_url,
                       service_key=settings.ai_service_key, max_retries=settings.callback_max_retries)
    except CallbackError:
        logger.error(f"Callback failed for {payload.requestId}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.server_host, port=settings.server_port)
