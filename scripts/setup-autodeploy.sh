#!/bin/bash
# Run this ONCE on your Hostinger server to enable auto-deploy.
# Usage: bash /opt/vesper/scripts/setup-autodeploy.sh

set -e

echo "=== Vesper Auto-Deploy Setup ==="

# Make deploy script executable
chmod +x /opt/vesper/scripts/deploy.sh

# Create systemd service for the webhook listener
cat > /etc/systemd/system/vesper-deploy.service << 'EOF'
[Unit]
Description=Vesper Deploy Webhook
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/vesper
ExecStart=/usr/bin/python3 /opt/vesper/scripts/deploy-webhook.py
Restart=always
RestartSec=5
Environment=DEPLOY_SECRET=vesper-deploy-2024

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
systemctl daemon-reload
systemctl enable vesper-deploy
systemctl start vesper-deploy

echo ""
echo "=== Done! ==="
echo ""
echo "Webhook is running on port 9000"
echo ""
echo "Next steps:"
echo "1. Go to GitHub repo → Settings → Webhooks → Add webhook"
echo "2. Payload URL: http://srv1438426.hstgr.cloud:9000/webhook"
echo "3. Content type: application/json"
echo "4. Secret: vesper-deploy-2024"
echo "5. Events: Just the push event"
echo ""
echo "After that, every git push will auto-deploy!"
