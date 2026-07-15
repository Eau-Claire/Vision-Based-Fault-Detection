# Deployment Plan: Vision-Based AI Service with SSL (Caddy CLI)

This guide outlines the production deployment of the Vision-Based AI Service (`server_pc`) under the domain name **`pms-ai.duckdns.org`** using Caddy to handle automatic SSL (HTTPS) provisioning entirely from the command line.

---

## 📋 Prerequisites

1. **Docker & Docker Compose**: `docker` and `docker-compose` or `docker compose` CLI.
2. **Ports Free**: Ensure ports **80** (HTTP) and **443** (HTTPS) are not in use by any other web server (like Nginx, Apache, or Nginx Proxy Manager).
3. **DuckDNS Configuration**: Update your `pms-ai.duckdns.org` domain to point to the host machine's public IP address.
4. **Firewall / Port Forwarding**: Ports **80** and **443** must be open and forwarded to this host machine.

---

## 🛠️ Step-by-Step Deployment

### 1. Release Ports 80 & 443
If Nginx Proxy Manager or another container is holding the ports, stop it:
```bash
docker stop npm
```

### 2. Configure the Environment
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

### 3. Deploy the Containers (RabbitMQ, AI Service, Caddy SSL)
Run the single command to pull, build, and start all services:
```bash
docker compose -f docker-compose.prod.yml up -d --build
```

---

## 🔍 Verification

Once the containers start up, Caddy will automatically negotiate SSL with Let's Encrypt for `pms-ai.duckdns.org`.

### 1. Check Container Health
```bash
docker compose -f docker-compose.prod.yml ps
```

### 2. Monitor SSL Provisioning Logs
To verify that Let's Encrypt successfully issued the SSL certificate:
```bash
docker logs -f ai_caddy_prod
```
*Look for logs saying: `authorization: awaiting challenge`, `validating challenge`, `certificate obtained successfully`.*

### 3. Test HTTPs Endpoint
Verify the setup using curl from any machine:
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