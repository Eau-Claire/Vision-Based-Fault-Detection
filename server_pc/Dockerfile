# =============================================================================
# Server PC — RF-DETR Dockerfile
# Supports NVIDIA CUDA GPU acceleration when available
# =============================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS gpu-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Python 3.11 and system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Copy requirements and install dependencies
COPY server_pc/requirements.txt /app/server_pc/requirements.txt
RUN pip install --no-cache-dir -r server_pc/requirements.txt

# Copy shared module
COPY shared/ /app/shared/

# Copy server module
COPY server_pc/ /app/server_pc/

# Create models directory
RUN mkdir -p /app/models/rf_detr

# Default environment variables
ENV DEVICE_PROFILE=server \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8002 \
    RFDETR_DEVICE=auto \
    LOG_FORMAT=json

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:8002/health'); r.raise_for_status()"

CMD ["python", "-m", "uvicorn", "server_pc.app.main:app", \
     "--host", "0.0.0.0", "--port", "8002"]
