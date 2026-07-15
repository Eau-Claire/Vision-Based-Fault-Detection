# Deployment Plan: Vision-Based AI Service with SSL

This guide outlines the production deployment of the Vision-Based AI Service (`server_pc`) under the domain name **`pms-ai.duckdns.org`** with automatic SSL (HTTPS) enabled using Caddy.

---

## 📋 Prerequisites

Before starting, ensure the host machine has the following installed:
1. **Docker & Docker Compose**: `docker` version 20.10+ and `docker-compose` or `docker compose` CLI.
2. **NVIDIA Container Toolkit** (Optional: only if GPU-accelerated inference is desired):
   - Installed and configured so Docker can access host GPUs (`nvidia-smi` works inside containers).
3. **Ports Open**: Ensure ports **80** (HTTP) and **443** (HTTPS) are open on your host firewall and forwarded in your router settings.
4. **DuckDNS Configuration**: Update your `pms-ai.duckdns.org` IP to point to the host machine's public IP address.

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

### 2. Verify Caddy Configuration
Make sure the `Caddyfile` is configured to listen on your domain:
```caddy
pms-ai.duckdns.org {
    reverse_proxy server_pc:8002

    log {
        output file /var/log/caddy/access.log
    }
}
```

### 3. Start the Production Services
Run the following command to build the image and start the containers in the background:
```bash
docker compose -f docker-compose.prod.yml up -d --build
```

---

## 🔍 Verification

Once deployment finishes, Caddy will automatically request and install a Let's Encrypt SSL certificate for your domain.

You can verify the status of the containers:
```bash
docker compose -f docker-compose.prod.yml ps
```

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

2. **Verify Caddy SSL logs**
   ```bash
   docker logs ai_caddy_prod
   ```

---

## 🐳 Container Architecture

```
Internet (pms-ai.duckdns.org)
       │ (Port 80 / 443 HTTPS)
       ▼
┌──────────────┐
│  Caddy Proxy │ (Auto Let's Encrypt SSL)
└──────┬───────┘
       │ (Internal Port 8002)
       ▼
┌──────────────┐       ┌──────────────┐
│  Server PC   ├──────►│   RabbitMQ   │
│  (FastAPI)   │       │   Broker     │
└──────────────┘       └──────────────┘
```