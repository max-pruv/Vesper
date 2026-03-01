#!/bin/bash
# Deploy Vesper to Hostinger VPS
# Usage: ./deploy.sh

set -e

HOST="root@srv1438426.hstgr.cloud"
KEY="$HOME/.ssh/id_ed25519"
REMOTE_DIR="/root/vesper"

echo "🚀 Deploying Vesper to production..."

# Push latest code to GitHub
echo ">>> Pushing to GitHub..."
git push origin main

# SSH deploy
echo ">>> Connecting to VPS..."
ssh -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -i "$KEY" "$HOST" << 'DEPLOY'
  cd /root/vesper
  echo ">>> Pulling latest code..."
  git fetch origin main
  git reset --hard origin/main
  echo ">>> Building Docker image..."
  docker compose build --no-cache
  echo ">>> Restarting services..."
  docker compose up -d
  echo ">>> Cleaning up old images..."
  docker image prune -f
  echo ">>> Deploy complete!"
  docker compose ps
DEPLOY

echo "✅ Deployment finished!"
