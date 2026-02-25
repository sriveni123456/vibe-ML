#!/bin/bash
# ═══════════════════════════════════════════
#  Vibe ML — Deploy Script
#  Run on a fresh Ubuntu 22.04 VPS
#  Usage: chmod +x deploy.sh && ./deploy.sh
# ═══════════════════════════════════════════

set -e
echo "═══ Vibe ML Deployment ═══"
echo ""

# ── 1. System Update ──
echo "→ Updating system..."
sudo apt update && sudo apt upgrade -y

# ── 2. Install Docker ──
if ! command -v docker &>/dev/null; then
    echo "→ Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
    echo "  Docker installed. You may need to logout/login for group changes."
else
    echo "→ Docker already installed."
fi

# ── 3. Install Docker Compose ──
if ! command -v docker-compose &>/dev/null; then
    echo "→ Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
else
    echo "→ Docker Compose already installed."
fi

# ── 4. Setup Environment ──
if [ ! -f .env ]; then
    echo "→ Creating .env from .env.example..."
    cp .env.example .env
    # Generate random JWT secret
    JWT_SECRET=$(openssl rand -hex 32)
    sed -i "s/CHANGE-THIS-to-a-random-64-character-string-in-production/$JWT_SECRET/" .env
    echo "  Generated JWT secret."
    echo ""
    echo "  ⚠️  IMPORTANT: Edit .env with your settings:"
    echo "     nano .env"
    echo ""
else
    echo "→ .env already exists."
fi

# ── 5. Create Storage Dirs ──
mkdir -p storage/uploads storage/outputs storage/models frontend

# ── 6. Build & Start ──
echo "→ Building Docker image..."
docker-compose build

echo "→ Starting services..."
docker-compose up -d

echo ""
echo "═══════════════════════════════════════════"
echo "  ✅ Vibe ML is running!"
echo ""
echo "  API:      http://$(hostname -I | awk '{print $1}'):8000"
echo "  Health:   http://$(hostname -I | awk '{print $1}'):8000/health"
echo "  Docs:     http://$(hostname -I | awk '{print $1}'):8000/docs"
echo ""
echo "  Next steps:"
echo "  1. Copy your frontend/index.html"
echo "  2. Point your domain to this server"
echo "  3. Setup SSL: sudo certbot --nginx -d vibeml.in"
echo "  4. Edit .env with Razorpay keys for payments"
echo "═══════════════════════════════════════════"
