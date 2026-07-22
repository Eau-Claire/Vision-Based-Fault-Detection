"""
Server PC — FastAPI application entry point.

Runs the server inference service through HarnessRuntime by default:
- /health and /ready endpoints
- RabbitMQ consumer for server analysis jobs
- REST API for ad-hoc analysis requests
"""

import os
import sys
import time
import threading
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from server_pc.app.settings import settings
from shared.utils.logging import (
    setup_logging,
    get_logger,
    set_correlation_context,
    clear_correlation_context,
)

setup_logging(
    level=settings.log_level,
    log_format=settings.log_format,
    service_name="server-pc",
)
logger = get_logger("main")

analysis_runner = None
_ready = False


def _init_analysis_runner():
    global analysis_runner, _ready
    try:
        from server_pc.app.analysis_runner import create_server_analysis_runner

        analysis_runner = create_server_analysis_runner(settings, Path(PROJECT_ROOT))
        _ready = True
        logger.info(
            f"{analysis_runner.model_name} analysis runner initialized",
            extra={
                "event": "analysis_runner_ready",
                "model": analysis_runner.model_name,
            },
        )

        # Start consumer now that the analysis runner is ready.
        if settings.rabbitmq_host:
            from server_pc.app.consumer import start_server_consumer
            start_server_consumer(analysis_runner, settings)
    except Exception as e:
        logger.error(f"Failed to init server analysis runner: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server PC service starting...")
    t = threading.Thread(target=_init_analysis_runner, daemon=True)
    t.start()
    yield
    logger.info("Server PC service shutting down...")


app = FastAPI(
    title="Server PC — AI Service",
    description=(
        "UAV Power-Line Inspection — Server inference using "
        "HarnessRuntime, Roboflow Workflow, or local RF-DETR"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

Path(settings.artifact_dir).mkdir(parents=True, exist_ok=True)
app.mount(
    settings.artifact_url_path,
    StaticFiles(directory=settings.artifact_dir),
    name="artifacts",
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "runtime": "server_pc",
        "backend": settings.inference_backend,
    }


@app.get("/ready")
def ready():
    if not _ready or analysis_runner is None:
        raise HTTPException(503, detail="Analysis runner not loaded yet")
    return {
        "status": "ready", "runtime": "server_pc",
        "model_name": analysis_runner.model_name,
        "model_version": analysis_runner.model_version,
    }


class AnalyzePayload(BaseModel):
    requestId: str
    mediaId: Optional[str] = None
    assetId: Optional[str] = None
    fileUrl: str
    mediaType: str = "Image"
    analysisType: str = "General"
    preferredModel: Optional[str] = None
    callbackUrl: Optional[str] = None
    correlationId: Optional[str] = None


@app.post("/api/analyze", status_code=202)
def analyze(payload: AnalyzePayload, background_tasks: BackgroundTasks):
    if not _ready or analysis_runner is None:
        raise HTTPException(503, detail="Analysis runner not loaded yet")
    background_tasks.add_task(_run_analysis, payload)
    return {"message": "Analysis started.", "requestId": payload.requestId, "runtime": "server_pc"}


def _run_analysis(payload: AnalyzePayload):
    from shared.services.callback_service import send_callback, CallbackError
    from shared.services.media_downloader import download_media, resolve_media_url
    from shared.services.result_mapper import map_success_result, map_failure_result

    set_correlation_context(
        correlation_id=payload.correlationId,
        request_id=payload.requestId,
    )
    start = time.monotonic()
    try:
        if _can_analyze_image_url(analysis_runner, payload.mediaType):
            media_url = resolve_media_url(
                file_url=payload.fileUrl,
                base_url=settings.callback_base_url,
                allow_private_ips=settings.allow_private_ips,
            )
            if not media_url.startswith("https://"):
                raise ValueError("Roboflow URL inputs must use https")
            output = analysis_runner.analyze_url(
                file_url=media_url,
                media_type=payload.mediaType,
            )
        else:
            file_bytes, ext = download_media(
                file_url=payload.fileUrl,
                base_url=settings.callback_base_url,
                timeout=settings.media_download_timeout,
                max_size_bytes=settings.media_max_size_bytes,
                allow_private_ips=settings.allow_private_ips,
            )
            output = analysis_runner.analyze_media(
                file_bytes=file_bytes,
                extension=ext,
                media_type=payload.mediaType,
                request_id=payload.requestId,
                artifact_dir=settings.artifact_dir,
                public_base_url=(
                    settings.artifact_public_base_url
                    or f"http://localhost:{settings.server_port}"
                ),
                artifact_url_path=settings.artifact_url_path,
                jpeg_quality=settings.artifact_jpeg_quality,
            )

        ms = int((time.monotonic() - start) * 1000)
        result = map_success_result(
            payload.requestId,
            payload.mediaId,
            output.detection_result,
            output.model_name,
            output.model_version,
            ms,
            "server",
            asset_id=payload.assetId,
            image_url=payload.fileUrl,
        )
        if output.harness_run_id:
            result.raw_result["harnessRunId"] = output.harness_run_id
        if output.harness_checkpoint_path:
            result.raw_result["harnessCheckpointPath"] = output.harness_checkpoint_path
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        result = map_failure_result(
            payload.requestId,
            payload.mediaId,
            "MODEL_INFERENCE_FAILED",
            str(e),
        )

    try:
        send_callback(
            result=result,
            callback_url=payload.callbackUrl or settings.callback_url,
            service_key=settings.ai_service_key,
            max_retries=settings.callback_max_retries,
            base_delay=settings.callback_retry_base_delay,
            max_delay=settings.callback_retry_max_delay,
            timeout=settings.callback_timeout,
            restrict_to_base_url=(
                settings.callback_base_url
                if settings.restrict_callback_to_base_url else None
            ),
            allow_private_ips=settings.allow_private_ips,
        )
    except CallbackError:
        logger.error(f"Callback failed for {payload.requestId}")
    finally:
        clear_correlation_context()


def _can_analyze_image_url(runner, media_type: str) -> bool:
    return media_type.lower() == "image" and hasattr(runner, "analyze_url")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.server_host, port=settings.server_port)
