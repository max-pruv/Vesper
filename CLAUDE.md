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
│    - vesper/dashboard/templates/ (Jinja2 templates)      │
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
| `vesper/dashboard/database.py` | SQLite user DB — auth, API keys (encrypted), password management |
| `vesper/dashboard/templates/dashboard.html` | Single-page dashboard — ~2400 lines HTML/JS |
| `vesper/dashboard/templates/trade_history.html` | Trade History page — KPI cards, win rate chart, filterable table |
| `vesper/dashboard/templates/settings.html` | Settings — API keys, trading config, change password, reset trades |
| `vesper/dashboard/templates/login.html` | Login — 2-step (hides 2FA when device trusted) |
| `vesper/dashboard/templates/how_it_works.html` | How Vesper works explainer |
| `vesper/dashboard/templates/admin.html` | Admin panel — LLM costs, users, bot state |
| `vesper/dashboard/templates/base.html` | Shared CSS/layout — glassmorphism dark theme |
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

### 7. JSON Null Safety — `_clean_nones()`
Portfolio JSON can contain `null` values. Python's `dict.get("key", 0)` returns `None` (not `0`) when the key exists with value `None`. This causes `TypeError` on arithmetic/comparisons. **FIXED globally**: `_clean_nones()` recursively replaces all `None` values with `0` at load time in `_load_portfolio()`. All portfolio data is guaranteed None-free after loading.

### 8. 2FA Login — Device Trust
Login is 2-step: if `vesper_trust` cookie exists (7-day device trust), the 2FA code field is hidden. If the cookie is expired/invalid, the POST handler redirects back with `?need_2fa=1` to show the field. Trust tokens are hashed (SHA-256) and stored in `trusted_devices` table.

## Pages & Navigation

All pages share a consistent nav bar:
**Dashboard | Markets | How it works | Trade History | Settings | [Admin] | Log out**

| Page | URL | Description |
|------|-----|-------------|
| Dashboard | `/dashboard` | Main autopilot management (2 tabs: Dashboard + Markets) |
| Markets | `/dashboard?tab=markets` | JS tab on dashboard — crypto, stocks, predictions sub-tabs |
| How it works | `/how-it-works` | System explainer |
| Trade History | `/trade-history` | Dedicated page — KPI cards, win rate chart, filterable table |
| Settings | `/settings` | API keys, trading config, change password, reset trades |
| Admin | `/admin` | LLM costs, API usage, user management |
| Login | `/login` | Email + password, conditional 2FA |

### Dashboard Tab (autopilot management)
- **Portfolio Overview** (top): Value, gradient chart (Total Value + Invested + Cash lines), stats grid
- **3 Autopilot Cards**: Crypto, Stocks, Predictions — each has ON/OFF state, fund/deployed/P&L display, single "Edit" button (opens unified modal with fund, max bet, risk, reinvest %)
- **Active Positions**: Cards with P&L sparkline, tooltips on all values, AI reasoning via `humanizeReason()`, Learn More modal
- **AI Decisions**: Feed with filter (all/entries/skips), human-readable descriptions

### Markets Tab (browsable market data — 3 sub-tabs)
- **Crypto**: Live prices for 60+ altcoins (ALTCOIN_UNIVERSE) — price, 24h%, volume, market cap (CoinGecko). Sortable columns, search filter.
- **Stocks**: US equities (STOCK_SYMBOLS) — live prices via Yahoo Finance. Price, 24h%, volume, market cap.
- **Predictions**: Polymarket/Kalshi browser with AI edge estimation, source selector, timeframe pills, methodology explainer.

### Trade History Page
- **4 KPI Cards**: Overall, Crypto, Stocks, Predictions win rates — color-coded green/red
- **Win Rate Chart**: Time-series line chart (Chart.js) — overall + per-vertical lines, 50% reference
- **Filters**: Type pills (All/Crypto/Stocks/Predictions) + date range inputs
- **Trade Table**: Date, Symbol, Type, Side, Entry/Exit Price, P&L ($/%%), Duration, Reason

### Settings Page
- **Exchange Connections**: Coinbase, Alpaca, Kalshi, Polymarket — connect/disconnect
- **AI Research**: Perplexity + Claude pipeline status
- **Trading**: Paper balance + "Reset All Trades" button (confirmation modal), risk management
- **Account**: Email, 2FA status, change password (complexity requirements)

### Key JS Functions
- `switchTab('trading'|'markets')` — main tab navigation
- `switchMarketTab('crypto'|'stocks'|'predictions')` — Markets sub-tab navigation
- `humanizeReason(raw)` — converts AI decision text to human-readable sentences
- `renderTopPortfolioChart()` — gradient area chart with 3 datasets
- `currentTradeMode` — initialized from server `{{ "real" if user.trading_mode == "live" else "paper" }}`

## API Endpoints (Key Ones)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/portfolio-stats` | GET | Live portfolio value, P&L, win rate |
| `/api/positions` | GET | Open positions with live prices |
| `/api/altcoin-hunter` | GET/POST | Crypto autopilot status/control |
| `/api/autopilot` | GET/POST | Stocks autopilot status/control |
| `/api/predictions-autopilot` | GET/POST | Predictions autopilot status/control |
| `/api/position-analysis/{pid}` | GET | Full AI analysis for Learn More modal |
| `/api/markets/crypto` | GET | Crypto market browser (60+ altcoins, live prices, market cap) |
| `/api/markets/stocks` | GET | Stock market browser (20 US equities, Yahoo Finance) |
| `/api/polymarket/markets` | GET | Polymarket prediction markets |
| `/api/kalshi/markets` | GET | Kalshi prediction markets |
| `/ws/live` | WS | Real-time updates (positions, stats, segments) |
| `/settings/change-password` | POST | Change user password (complexity validated) |
| `/settings/reset-trades` | POST | Full portfolio reset — clears everything |
| `/api/admin/bot-state` | GET | Debug: user configs, scan logs |
| `/api/admin/scan-test` | GET | Debug: test exchange + scanning pipeline |
| `/api/admin/deploy` | POST | Hot-deploy: git pull + restart |
| `/api/health` | GET | Health check with cycle state |

## Common Issues & Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| All crypto scans fail | Binance geo-blocked (HTTP 451) | Already fixed — auto-exchange selection |
| Portfolio value shows $5 | Using `cash` instead of total funds | Fixed — includes autopilot fund_usd |
| 500 errors on API endpoints | JSON null → `.get("key", 0)` returns None | Fixed — `_clean_nones()` at load time |
| Active positions not showing | trade_mode filter mismatch | Fixed — server-side init + fallback |
| Portfolio chart not loading | Seed point timing issue | Fixed — seed 30s in the past |
| AI decisions incomprehensible | Raw strategy output shown directly | Fixed — `humanizeReason()` |
| Start button "$10 minimum" error | Input empty, parseFloat("") = NaN | Fixed — fallback to placeholder value |
| Docker `--no-cache` flag error | Wrong command syntax | Use `docker compose build --no-cache && docker compose up -d` |
| Docker network conflict | `network already exists` | `docker compose down --remove-orphans && docker network prune -f` |
| Nav menu inconsistent | Each page had its own nav | Fixed — all pages share consistent nav order |

## Logging

Comprehensive logging via `_log = logging.getLogger("vesper.dashboard")`:
- Login attempts (success/fail/locked/2FA)
- Dashboard loads (user, mode, position count, trade count)
- API endpoint calls (portfolio-stats, positions, altcoin-hunter)
- WebSocket connections/disconnections
- Password changes, trade resets
- All errors with full tracebacks

## Deployment

```bash
# On local machine: push to branch
git push -u origin claude/deploy-openclaw-cloudflare-GkBQL

# On server (SSH): pull and rebuild
cd /opt/vesper
git pull origin claude/deploy-openclaw-cloudflare-GkBQL
docker compose down --remove-orphans && docker network prune -f && docker compose up -d --build

# Force rebuild (no cache):
docker compose build --no-cache && docker compose up -d

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
- **Dashboard tab**: Real-time via WebSocket + 10s polling fallback. Portfolio gradient chart, tooltips, unified Edit modal, humanized AI decisions.
- **Markets tab**: 3 sub-tabs (Crypto, Stocks, Predictions). Crypto shows live prices for ALTCOIN_UNIVERSE + market cap via CoinGecko. Stocks via Yahoo Finance.
- **Trade History page**: Dedicated page with KPI cards per vertical, win rate time-series chart, filterable table (by type + date range).
- **Settings page**: Exchange connections, trading config, change password (complexity requirements), reset all trades (confirmation modal).
- **Login**: 2-step — hides 2FA field when device is trusted (7-day cookie).
- **Nav**: Consistent across all pages — Dashboard, Markets, How it works, Trade History, Settings, [Admin], Log out.
- **Logging**: Comprehensive — login, API calls, WebSocket, errors with tracebacks.
- **NoneType safety**: `_clean_nones()` at portfolio load time prevents all null-related crashes.
- **Price recording**: Every 60s per position for sparkline charts.
- **currentTradeMode**: Initialized from server-side `user.trading_mode` (not hardcoded 'paper').
- **Hot-deploy**: `/api/admin/deploy` does git pull + restart. Git installed in Dockerfile.
