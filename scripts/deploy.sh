#!/bin/bash
# Vesper auto-deploy script
# Called by the webhook or manually

set -e

REPO_DIR="/opt/vesper"
BRANCH="main"
LOG="/opt/vesper/deploy.log"

echo "$(date) — Deploy triggered" >> "$LOG"

cd "$REPO_DIR"

# Pull latest
git fetch origin "$BRANCH" 2>>"$LOG"
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse "origin/$BRANCH")

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "$(date) — Already up to date ($LOCAL)" >> "$LOG"
    exit 0
fi

echo "$(date) — Updating $LOCAL → $REMOTE" >> "$LOG"
git pull origin "$BRANCH" >> "$LOG" 2>&1

# Rebuild and restart
docker compose down >> "$LOG" 2>&1
docker compose up -d --build >> "$LOG" 2>&1

echo "$(date) — Deploy complete" >> "$LOG"
