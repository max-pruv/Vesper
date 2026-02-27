#!/bin/bash
# Vesper — Hostinger VPS Setup Script
# Run this on your VPS: bash deploy/hostinger_setup.sh

set -e

echo "=========================================="
echo "  Vesper — Hostinger VPS Setup"
echo "=========================================="

# Update system
echo "[1/5] Updating system..."
apt-get update -y && apt-get upgrade -y

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "[2/5] Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
else
    echo "[2/5] Docker already installed"
fi

# Install Docker Compose plugin if not present
if ! docker compose version &> /dev/null; then
    echo "[3/5] Installing Docker Compose..."
    apt-get install -y docker-compose-plugin
else
    echo "[3/5] Docker Compose already installed"
fi

# Clone repo if not exists
REPO_DIR="/opt/vesper"
if [ ! -d "$REPO_DIR" ]; then
    echo "[4/5] Cloning Vesper repository..."
    git clone https://github.com/max-pruv/Vesper.git "$REPO_DIR"
else
    echo "[4/5] Updating Vesper repository..."
    cd "$REPO_DIR" && git pull origin main
fi

cd "$REPO_DIR"

# Create .env if not exists
if [ ! -f ".env" ]; then
    echo "[5/5] Creating .env from template..."
    cp .env.example .env
    echo ""
    echo "=========================================="
    echo "  IMPORTANT: Edit your .env file!"
    echo "=========================================="
    echo "  nano /opt/vesper/.env"
    echo ""
    echo "  Set your Coinbase API credentials:"
    echo "    COINBASE_API_KEY=..."
    echo "    COINBASE_API_SECRET=..."
    echo ""
    echo "  Then start the bot:"
    echo "    cd /opt/vesper"
    echo "    docker compose up -d"
    echo "=========================================="
else
    echo "[5/5] .env already exists"
    echo ""
    echo "Starting Vesper..."
    docker compose up -d --build
    echo ""
    echo "=========================================="
    echo "  Vesper is running!"
    echo "  Logs: docker compose logs -f"
    echo "=========================================="
fi
