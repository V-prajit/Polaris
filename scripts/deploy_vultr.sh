#!/usr/bin/env bash
set -euo pipefail
echo "=== ParkSight Vultr Deployment ==="

# System deps
apt update && apt install -y python3.11 python3.11-venv python3-pip curl git docker.io docker-compose

# Node.js 20 LTS
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs

# Repo should already be cloned to /opt/parksight
cd /opt/parksight

# Python venv
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt

# Env config — first run prompts user to edit
if [ ! -f .env ]; then
    cp .env.production .env
    VULTR_IP=$(curl -s ifconfig.me)
    sed -i "s/YOUR_VULTR_PUBLIC_IP/$VULTR_IP/g" .env
    echo ""
    echo ">>> .env created with IP: $VULTR_IP"
    echo ">>> EDIT .env NOW: add your GEMINI_API_KEY and GOOGLE_MAPS_API_KEY"
    echo ">>> Then re-run this script."
    exit 1
fi

# Start VectorDB container
docker-compose up -d

# Build frontend (reads NEXT_PUBLIC_API_URL from .env)
set -a && source .env && set +a
npm install && npm run build

# Systemd: API backend
cat > /etc/systemd/system/parksight-api.service << 'EOF'
[Unit]
Description=ParkSight FastAPI Backend
After=network.target docker.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/parksight
EnvironmentFile=/opt/parksight/.env
ExecStart=/opt/parksight/venv/bin/uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Systemd: Next.js frontend
cat > /etc/systemd/system/parksight-web.service << 'EOF'
[Unit]
Description=ParkSight Next.js Frontend
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/parksight
ExecStart=/usr/bin/npm start -- -p 3000
Restart=always
RestartSec=5
Environment=NODE_ENV=production

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable parksight-api parksight-web
systemctl start parksight-api parksight-web

# Firewall
ufw allow 22/tcp && ufw allow 3000/tcp && ufw allow 8000/tcp && ufw --force enable

sleep 3
echo ""
echo "=== Deployment Complete ==="
echo "Frontend: http://$(curl -s ifconfig.me):3000"
echo "API:      http://$(curl -s ifconfig.me):8000/api/health"
echo "VectorDB: localhost:50051 (internal only)"
