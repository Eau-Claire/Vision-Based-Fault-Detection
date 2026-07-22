# =============================================================================
# Edge Raspberry Pi — YOLO11 Dockerfile
# Multi-arch: supports linux/arm64 (Raspberry Pi) and linux/amd64
# =============================================================================
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for OpenCV headless
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY edge_raspberry/requirements.txt /app/edge_raspberry/requirements.txt
RUN pip install --no-cache-dir -r edge_raspberry/requirements.txt

# Copy shared module
COPY shared/ /app/shared/

# Copy edge module
COPY edge_raspberry/ /app/edge_raspberry/

# Create models directory
RUN mkdir -p /app/models/yolo11 /app/artifacts

# Default environment variables
ENV DEVICE_PROFILE=edge \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8001 \
    YOLO_DEVICE=cpu \
    AI_ARTIFACT_DIR=/app/artifacts \
    AI_ARTIFACT_URL_PATH=/artifacts \
    LOG_FORMAT=json

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:8001/health'); r.raise_for_status()"

CMD ["python", "-m", "uvicorn", "edge_raspberry.app.main:app", \
     "--host", "0.0.0.0", "--port", "8001"]
