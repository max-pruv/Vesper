#!/bin/bash
# Vesper auto-deploy script
# Called by the webhook or manually

set -e

REPO_DIR="/opt/vesper"
LOG="/opt/vesper/deploy.log"
DEPLOY_BRANCH="claude/deploy-openclaw-cloudflare-GkBQL"

cd "$REPO_DIR"

echo "$(date) — Deploy triggered" >> "$LOG"

# Ensure we're on the correct branch
CURRENT=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT" != "$DEPLOY_BRANCH" ]; then
    echo "$(date) — Switching from $CURRENT to $DEPLOY_BRANCH" >> "$LOG"
    git fetch origin "$DEPLOY_BRANCH" 2>>"$LOG"
    git checkout "$DEPLOY_BRANCH" >> "$LOG" 2>&1
fi

# Pull latest
git fetch origin "$DEPLOY_BRANCH" 2>>"$LOG"
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse "origin/$DEPLOY_BRANCH")

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "$(date) — Already up to date ($LOCAL)" >> "$LOG"
    exit 0
fi

echo "$(date) — Updating $LOCAL → $REMOTE" >> "$LOG"
git pull origin "$DEPLOY_BRANCH" >> "$LOG" 2>&1

# Rebuild and restart
docker compose down >> "$LOG" 2>&1
docker compose up -d --build >> "$LOG" 2>&1

echo "$(date) — Deploy complete" >> "$LOG"
