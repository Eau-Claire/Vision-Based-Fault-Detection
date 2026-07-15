# Deployment Plan: Vision-Based AI Service with SSL (Nginx Proxy Manager)

This guide outlines the production deployment of the Vision-Based AI Service (`server_pc`) under the domain name **`pms-ai.duckdns.org`** with automatic SSL (HTTPS) enabled using your existing **Nginx Proxy Manager**.

---

## 📋 Prerequisites

Before starting, ensure the host machine has the following:
1. **Docker & Docker Compose**: `docker` version 20.10+ and `docker-compose` or `docker compose` CLI.
2. **Ports Open**: Port **8002** (FastAPI) is mapped from the container to the host.
3. **DuckDNS Configuration**: Update your `pms-ai.duckdns.org` domain to point to the host machine's public IP address.
4. **Nginx Proxy Manager (NPM)**: Running on ports 80 and 443 (already confirmed).

---

## 🛠️ Step-by-Step Deployment

### 1. Configure the Environment
Ensure your `.env` file in the root directory contains the production-ready keys:
```env
DEVICE_PROFILE=server
ROBOFLOW_API_KEY=3VRKN4GLeDVKPJ9eQFkG
RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASS=guest
AI_SERVICE_KEY=Reme8lqiErnO9ZppU0SeNattf4ObRvbv
CALLBACK_BASE_URL=https://uavpms.ddns.net
```

### 2. Start the Production Services
Run the following command to build the image and start the containers in the background:
```bash
docker compose -f docker-compose.prod.yml up -d --build
```
*Note: This will start RabbitMQ and the AI service server, exposing port `8002` to the host.*

### 3. Route Traffic through Nginx Proxy Manager (SSL)
Open your Nginx Proxy Manager Dashboard (usually on port `81`, e.g., `http://your-server-ip:81`) and add a new **Proxy Host**:

1. **Details Tab**:
   - **Domain Names**: `pms-ai.duckdns.org`
   - **Scheme**: `http`
   - **Forward Hostname / IP**: `127.0.0.1` (or the local IP of the host machine)
   - **Forward Port**: `8002`
   - **Block Common Exploits**: Enabled
   - **Websockets Support**: Enabled (optional)

2. **SSL Tab**:
   - **SSL Certificate**: Select **Request a new SSL Certificate** (Let's Encrypt)
   - **Force SSL**: Enabled
   - **HTTP/2 Support**: Enabled
   - **I Agree to the Let's Encrypt Terms of Service**: Enabled
   - Save the host!

---

## 🔍 Verification

Once Nginx Proxy Manager finishes requesting the certificate:

### Test API Connectivity

1. **Verify Readiness Endpoint (HTTPS)**
   ```bash
   curl -i https://pms-ai.duckdns.org/ready
   ```
   **Expected Response:**
   ```json
   HTTP/2 200
   content-type: application/json
   ...
   {"status":"ready","runtime":"server_pc","model_name":"RF-DETR-Base","model_version":"1.0.0"}
   ```