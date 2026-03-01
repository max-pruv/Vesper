# Vesper — AI Trading Bot Platform

## Quick Start for New Claude Conversations

**Branch:** `claude/deploy-openclaw-cloudflare-GkBQL`
**Server:** Hostinger VPS at `srv1438426.hstgr.cloud`
**Deploy:** Push to branch, then trigger webhook on server port 9876, OR SSH and run:
```bash
cd /opt/vesper && git pull origin claude/deploy-openclaw-cloudflare-GkBQL && docker compose down && docker compose up -d --build
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Caddy (HTTPS reverse proxy) → port 80/443              │
│    ↓                                                     │
│  FastAPI Dashboard (port 8080)                           │
│    - vesper/dashboard/app.py (backend)                   │
│    - vesper/dashboard/templates/dashboard.html (frontend)│
│    - WebSocket at /ws/live (2s updates)                  │
│    - REST polling fallback (10s)                         │
│    ↓                                                     │
│  APScheduler (BlockingScheduler, every 1 minute)         │
│    → vesper/main.py: Vesper.run_all_users()              │
│    → UserBot.run_cycle() per active user                 │
│                                                          │
│  3 Verticals:                                            │
│    1. Crypto (Altcoin Hunter) — 60+ coins, trend scoring │
│    2. Stocks (Autopilot) — via Alpaca API                │
│    3. Predictions — Polymarket + Kalshi, AI research     │
│                                                          │
│  AI Pipeline: Perplexity Sonar → Claude Haiku            │
│  Exchange: Auto-selected (BinanceUS/KuCoin/OKX/Bybit)   │
│  Data: JSON files per user at /app/data/portfolio_*.json │
└─────────────────────────────────────────────────────────┘
```

## Critical Files

| File | Purpose |
|------|---------|
| `vesper/main.py` | Main orchestrator — `UserBot` class, all scanning functions, scheduler |
| `vesper/dashboard/app.py` | FastAPI backend — all API endpoints, WebSocket, price fetching |
| `vesper/dashboard/templates/dashboard.html` | Single-page dashboard — ~2200 lines HTML/JS |
| `vesper/state.py` | Shared runtime state (solves `__main__` vs `vesper.main` module split) |
| `vesper/strategies/altcoin_hunter.py` | 6-factor trend scoring (momentum, RSI, EMA, volume, ADX, BTC relative) |
| `vesper/ai_research.py` | Perplexity + Claude pipeline for deep market research |
| `vesper/polymarket.py` | Polymarket integration via Gamma API |
| `vesper/kalshi.py` | Kalshi markets integration |
| `vesper/market_data.py` | Technical indicators, multi-timeframe snapshots |
| `vesper/portfolio.py` | Portfolio manager — positions, trades, JSON persistence |
| `config/settings.py` | Ticker symbols, altcoin universe, stock symbols |
| `docker-compose.yml` | Docker deployment (Caddy + Vesper) |
| `deploy-webhook.py` | Deploy webhook on port 9876 |

## Key Technical Decisions (Don't Redo These)

### 1. Binance is Geo-Blocked (HTTP 451)
The Hostinger VPS is in a restricted country. We auto-select exchanges:
```
_EXCHANGE_CANDIDATES: Binance → BinanceUS → KuCoin → OKX → Bybit
```
**BinanceUS works.** The `_resolve_exchange()` function in `main.py` handles this with a cached singleton.

### 2. Thread-Safe Exchange Instances
Parallel scanning uses `ThreadPoolExecutor`. Each thread gets its own `ccxt` exchange instance via `threading.local()` with shared markets cache. See `_get_thread_exchange()`.

### 3. Python Module Split (`__main__` vs `vesper.main`)
When running `python -m vesper.main`, the module is `__main__`, but imports reference `vesper.main` — creating TWO separate module instances. Shared state lives in `vesper/state.py`.

### 4. Dashboard Price Fetching
`_get_public_exchange()` in `app.py` uses `_resolve_exchange()` from main.py (NOT raw `ccxt.binance()` which would fail due to geo-blocking).

### 5. Portfolio Value = cash + ALL autopilot funds + unrealized P&L
Not just `cash` (which is leftover unallocated money, typically ~$5). The autopilot funds are stored in:
- `portfolio["altcoin_hunter"]["fund_usd"]`
- `portfolio["autopilot"]["fund_usd"]`
- `portfolio["predictions_autopilot"]["fund_usd"]`

### 6. Position Price History
Recorded every cycle (~60s) by `_record_price_snapshots()` in `UserBot.run_cycle()`. Also recorded by WS handler when client is connected. Stored in `position["price_history"]` as `[{t, p, w}, ...]`.

## Dashboard UI Structure

- **Portfolio Overview** (top): Value, chart (stacked invested/uninvested), stats grid
- **3 Autopilot Cards**: Crypto, Stocks, Predictions — each has ON/OFF state, fund/deployed/P&L display, single "Edit" button (opens unified modal with fund, max bet, risk, reinvest %)
- **Active Positions**: Cards with P&L sparkline, tooltips on all values, AI reasoning, Learn More modal
- **P&L section removed** — replaced by portfolio chart at top
- **Trade History**: Table of closed trades
- **Predictions Tab**: Polymarket/Kalshi browser with AI edge estimation

## API Endpoints (Key Ones)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/portfolio-stats` | GET | Live portfolio value, P&L, win rate |
| `/api/positions` | GET | Open positions with live prices |
| `/api/altcoin-hunter` | GET/POST | Crypto autopilot status/control |
| `/api/autopilot` | GET/POST | Stocks autopilot status/control |
| `/api/predictions-autopilot` | GET/POST | Predictions autopilot status/control |
| `/api/position-analysis/{pid}` | GET | Full AI analysis for Learn More modal |
| `/ws/live` | WS | Real-time updates (positions, stats, segments) |
| `/api/admin/bot-state` | GET | Debug: user configs, scan logs |
| `/api/admin/scan-test` | GET | Debug: test exchange + scanning pipeline |
| `/api/health` | GET | Health check with cycle state |

## Common Issues & Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| All crypto scans fail | Binance geo-blocked (HTTP 451) | Already fixed — auto-exchange selection |
| Portfolio value shows $5 | Using `cash` instead of total funds | Fixed — includes autopilot fund_usd |
| Predictions "waiting for scan" | Scan errors not logged | Fixed — error logging added |
| P&L chart flat/empty | No price history recorded | Fixed — `_record_price_snapshots()` in scheduler |
| `EnhancedEnsemble` not defined | Missing import | Fixed — added to top-level imports |
| Cycle count = 0 | `__main__` vs `vesper.main` split | Fixed — shared `vesper/state.py` |
| Prices show $0 | Dashboard used `ccxt.binance()` directly | Fixed — uses `_resolve_exchange()` |

## Deployment

```bash
# On local machine: push to branch
git push -u origin claude/deploy-openclaw-cloudflare-GkBQL

# On server (SSH): pull and rebuild
cd /opt/vesper
git pull origin claude/deploy-openclaw-cloudflare-GkBQL
docker compose down && docker compose up -d --build

# Check logs
docker logs vesper-bot --tail 50 -f

# Check health
curl -s http://localhost:8080/api/health | python3 -m json.tool
curl -s http://localhost:8080/api/admin/bot-state | python3 -m json.tool
```

## Current State (as of latest commit)

- **Crypto autopilot**: Working. BinanceUS auto-selected. Scans 60+ coins, opens positions.
- **Stocks autopilot**: Needs Alpaca API keys configured by user.
- **Predictions autopilot**: Should work if Perplexity + Anthropic keys are set. Runs hourly.
- **Dashboard**: Real-time via WebSocket + 10s polling fallback. Portfolio chart, tooltips, unified Edit modal.
- **Price recording**: Every 60s per position for sparkline charts.
