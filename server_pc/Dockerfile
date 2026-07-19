# =============================================================================
# Server PC Dockerfile
# Default image is CPU/harness friendly. Local RF-DETR dependencies are optional.
# =============================================================================
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY server_pc/requirements.txt /app/server_pc/requirements.txt
RUN pip install --no-cache-dir -r server_pc/requirements.txt

ARG INSTALL_LOCAL_ML=false
COPY server_pc/requirements-local.txt /app/server_pc/requirements-local.txt
RUN if [ "$INSTALL_LOCAL_ML" = "true" ]; then \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
            torch==2.3.1+cpu torchvision==0.18.1+cpu && \
        pip install --no-cache-dir -r server_pc/requirements-local.txt; \
    fi

COPY shared/ /app/shared/
COPY edge/ /app/edge/
COPY server_pc/ /app/server_pc/

RUN mkdir -p /app/models/rf_detr /tmp/vision-harness-checkpoints

ENV DEVICE_PROFILE=server \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8002 \
    SERVER_INFERENCE_BACKEND=harness \
    HARNESS_CHECKPOINT_DIR=/tmp/vision-harness-checkpoints \
    RFDETR_DEVICE=cpu \
    LOG_FORMAT=json

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:8002/health'); r.raise_for_status()"

CMD ["python", "-m", "uvicorn", "server_pc.app.main:app", \
     "--host", "0.0.0.0", "--port", "8002"]
