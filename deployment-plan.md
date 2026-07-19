# Deployment Plan: Vision-Based AI Service with Host Nginx & Certbot

This guide outlines how to deploy the Vision-Based AI Service (`server_pc`) on port `8002` and configure your host's existing **Nginx** server to reverse proxy the subdomain **`pms-ai.duckdns.org`** with HTTPS/SSL using **Certbot**.

---

## 🛠️ Step-by-Step Deployment

### 1. Build and Start the AI Service
Run the following command in the root folder of the project on your server to pull, build, and start the AI service. The FastAPI application will run on port `8002` (bound to `127.0.0.1` for security so only Nginx can access it):
```bash
git pull origin main
docker compose -f docker-compose.prod.yml up -d --build server_pc
```

The production Dockerfile defaults to the lightweight CPU/harness backend and does not install RF-DETR/PyTorch. Use `INSTALL_LOCAL_ML=true SERVER_INFERENCE_BACKEND=local` only when intentionally deploying the optional local RF-DETR backend.

### 2. Configure Host Nginx
Create a new configuration file for the AI service in your host's Nginx configuration directory:
```bash
sudo nano /etc/nginx/sites-available/pms-ai.duckdns.org
```

Paste the following Nginx configuration inside:
```nginx
server {
    listen 80;
    server_name pms-ai.duckdns.org;

    location / {
        proxy_pass http://127.0.0.1:8002;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable the configuration by creating a symlink to `sites-enabled`:
```bash
sudo ln -s /etc/nginx/sites-available/pms-ai.duckdns.org /etc/nginx/sites-enabled/
```

Test the Nginx configuration for syntax errors and restart Nginx:
```bash
sudo nginx -t
sudo systemctl restart nginx
```

### 3. Generate SSL Certificate with Certbot
Run Certbot on the host to automatically request, download, and configure the SSL certificate for your subdomain:
```bash
sudo certbot --nginx -d pms-ai.duckdns.org
```
*Note: Certbot will ask if you want to redirect HTTP traffic to HTTPS. Select **2** (Redirect) to enforce SSL.*

---

## 🔍 Verification

Verify that the secure HTTPS endpoint is reachable and returning a status of ready:
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