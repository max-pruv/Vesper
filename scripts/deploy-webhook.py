#!/usr/bin/env python3
"""Tiny GitHub webhook receiver for auto-deploy.

Runs on port 9000 outside Docker. When GitHub sends a push event,
it verifies the secret and runs deploy.sh.

Usage:
    DEPLOY_SECRET=your_secret python3 deploy-webhook.py
"""

import hashlib
import hmac
import json
import os
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler

SECRET = os.environ.get("DEPLOY_SECRET", "vesper-deploy-2024")
DEPLOY_SCRIPT = "/opt/vesper/scripts/deploy.sh"
PORT = 9000


class WebhookHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/webhook":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        # Verify GitHub signature
        signature = self.headers.get("X-Hub-Signature-256", "")
        if signature:
            expected = "sha256=" + hmac.new(
                SECRET.encode(), body, hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(signature, expected):
                self.send_response(403)
                self.end_headers()
                self.wfile.write(b"Invalid signature")
                return

        # Parse event
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        event = self.headers.get("X-GitHub-Event", "ping")

        if event == "ping":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"pong")
            return

        if event == "push":
            ref = payload.get("ref", "")
            print(f"Push event on {ref}")

            # Run deploy in background
            subprocess.Popen(
                ["bash", DEPLOY_SCRIPT],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Deploying...")
            return

        self.send_response(200)
        self.end_headers()
        self.wfile.write(f"Ignored event: {event}".encode())

    def log_message(self, format, *args):
        print(f"[webhook] {args[0]}")


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), WebhookHandler)
    print(f"Webhook listener on port {PORT}")
    print(f"URL: http://0.0.0.0:{PORT}/webhook")
    server.serve_forever()
