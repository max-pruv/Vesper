#!/usr/bin/env python3
"""Lightweight deploy webhook — listens on port 9876.

Triggered via:
    curl -X POST https://<server-ip>:9876/deploy -H "Authorization: Bearer <TOKEN>"

Pulls latest code and rebuilds Docker containers.
"""

import subprocess
import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading

DEPLOY_TOKEN = os.environ.get("DEPLOY_TOKEN", "")
if not DEPLOY_TOKEN:
    raise RuntimeError("DEPLOY_TOKEN environment variable is required")
REPO_DIR = "/opt/vesper"
BRANCH = "claude/deploy-openclaw-cloudflare-GkBQL"
PORT = 9876

deploy_lock = threading.Lock()
last_deploy = {"time": 0, "status": "", "output": ""}


class DeployHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/deploy":
            self.send_response(404)
            self.end_headers()
            return

        # Auth check
        auth = self.headers.get("Authorization", "")
        if auth != f"Bearer {DEPLOY_TOKEN}":
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b'{"error":"forbidden"}')
            return

        if not deploy_lock.acquire(blocking=False):
            self.send_response(409)
            self.end_headers()
            self.wfile.write(b'{"error":"deploy already in progress"}')
            return

        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "deploying"}).encode())

            # Run deploy in background thread
            threading.Thread(target=_run_deploy, daemon=True).start()
        except Exception:
            deploy_lock.release()

    def do_GET(self):
        if self.path == "/deploy/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(last_deploy).encode())
        elif self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'ok')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        print(f"[deploy-webhook] {args[0]}")


def _run_deploy():
    global last_deploy
    try:
        cmds = [
            f"cd {REPO_DIR} && git fetch origin {BRANCH}",
            f"cd {REPO_DIR} && git checkout {BRANCH}",
            f"cd {REPO_DIR} && git pull origin {BRANCH}",
            f"cd {REPO_DIR} && docker compose down",
            f"cd {REPO_DIR} && docker compose up -d --build",
        ]
        output = ""
        for cmd in cmds:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            output += f"$ {cmd}\n{result.stdout}\n"
            if result.returncode != 0:
                output += f"STDERR: {result.stderr}\n"
                last_deploy = {"time": time.time(), "status": "error", "output": output}
                return

        last_deploy = {"time": time.time(), "status": "success", "output": output}
        print("[deploy-webhook] Deploy completed successfully")
    except Exception as e:
        last_deploy = {"time": time.time(), "status": "error", "output": str(e)}
    finally:
        deploy_lock.release()


if __name__ == "__main__":
    print(f"[deploy-webhook] Listening on port {PORT}")
    server = HTTPServer(("0.0.0.0", PORT), DeployHandler)
    server.serve_forever()
