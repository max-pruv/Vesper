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
│  Learning: Self-adaptive strategy optimization           │
└─────────────────────────────────────────────────────────┘
```

## Critical Files

| File | Purpose |
|------|---------|
| `vesper/main.py` | Main orchestrator — `UserBot` class, all scanning functions, scheduler, prediction price refresh, learning engine integration |
| `vesper/dashboard/app.py` | FastAPI backend — all API endpoints, WebSocket, price fetching, onboarding, live position analysis |
| `vesper/dashboard/database.py` | SQLite user DB — auth, API keys (encrypted), password management, onboarding_complete flag |
| `vesper/dashboard/templates/dashboard.html` | Single-page dashboard — ~2400 lines HTML/JS |
| `vesper/dashboard/templates/onboarding.html` | 4-step onboarding wizard — interests, exchange connections, trading mode |
| `vesper/dashboard/templates/trade_history.html` | Trade History page — KPI cards, win rate chart, filterable table |
| `vesper/dashboard/templates/settings.html` | Settings — API keys, trading config, change password, reset trades |
| `vesper/dashboard/templates/login.html` | Login — 2-step (hides 2FA when device trusted) |
| `vesper/dashboard/templates/how_it_works.html` | How Vesper works explainer |
| `vesper/dashboard/templates/admin.html` | Admin panel — LLM costs, users, bot state |
| `vesper/dashboard/templates/base.html` | Shared CSS/layout — glassmorphism dark theme (CSS vars, Inter font) |
| `vesper/state.py` | Shared runtime state (solves `__main__` vs `vesper.main` module split) |
| `vesper/learning.py` | Self-learning engine — analyzes trade history, adjusts thresholds/SL/TP/sizing |
| `vesper/strategies/altcoin_hunter.py` | 6-factor trend scoring (momentum, RSI, EMA, volume, ADX, BTC relative) |
| `vesper/ai_research.py` | Perplexity + Claude pipeline for deep market research |
| `vesper/social_sentiment.py` | X/Twitter social sentiment analysis |
| `vesper/polymarket.py` | Polymarket integration via Gamma API + microstructure edge estimation |
| `vesper/kalshi.py` | Kalshi markets integration (CFTC-regulated prediction markets) |
| `vesper/market_data.py` | Technical indicators, multi-timeframe snapshots, order book pressure |
| `vesper/portfolio.py` | Portfolio manager — positions, trades, JSON persistence, preserves extra fields |
| `vesper/risk.py` | Risk management — SL/TP/trailing stop, position sizing, ATR-based dynamic SL |
| `config/settings.py` | Ticker symbols, altcoin universe, stock symbols, exchange/risk configs |
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

### 9. Prediction Position Price Refresh
`_refresh_prediction_prices()` runs every cycle (~60s). Fetches current YES/NO prices from Polymarket Gamma API + Kalshi REST API. Updates `current_probability` in raw position JSON. Checks SL/TP limits and auto-closes positions when hit. This makes prediction P&L live on dashboard (was previously showing stale entry price).

### 10. Portfolio._save_state() Preserves Extra Fields
`_save_state()` rebuilds position dicts from Position dataclass. Extra fields written by other modules (`price_history`, `est_fee`, `current_probability`, `prediction_question`, `prediction_side`, `prediction_ai_prob`, `prediction_mkt_prob`, `prediction_edge`) are carried over from the existing raw JSON via `_PRESERVED_POS_KEYS`. Without this, these fields would be lost on every position open/close.

### 11. Self-Learning Algorithm (vesper/learning.py)
Runs every 10 minutes via `_update_learning_state()` in `run_cycle()`. Requires 8+ closed trades to activate. Analyzes rolling window of 50 trades per strategy and adjusts:
- **Entry threshold** (±15% max): raises threshold when win rate < 35%, lowers when > 60%
- **Stop-loss** (±30% max): tightens when too many SL exits + low win rate
- **Take-profit** (±30% max): widens when avg win < avg loss
- **Position sizing** (0.5x-1.3x): scales with profit factor
- **Rebalance threshold**: adjusts when to exit weakening altcoin positions
All adjustments are smoothed (30% per cycle) and decay toward defaults when neutral. State stored in `portfolio["learning_state"]`.

### 12. Learn More — Live Deep Research
`/api/position-analysis/{pid}` returns full AI analysis for a position. If no pre-saved analysis exists (e.g., for positions opened before analysis saving was added), it runs **live deep research on-demand** using Perplexity + Claude + social sentiment. The result is cached in `position_analyses` for future requests.

### 13. Onboarding Wizard
New users are redirected to `/onboarding` (4-step wizard) until they complete it. Existing users with any activity (positions, trades, API keys, enabled autopilots) are auto-marked as onboarding_complete and skip it. The wizard allows connecting exchange accounts as an optional step. `onboarding_complete` flag stored in SQLite `users` table.

## Pages & Navigation

All pages share a consistent nav bar:
**Dashboard | Markets | How it works | Trade History | Settings | [Admin] | Log out**

| Page | URL | Description |
|------|-----|-------------|
| Onboarding | `/onboarding` | 4-step setup wizard (Welcome → Interests → Connections → Summary) |
| Dashboard | `/dashboard` | Main autopilot management (2 tabs: Dashboard + Markets) |
| Markets | `/dashboard?tab=markets` | JS tab on dashboard — crypto, stocks, predictions sub-tabs |
| How it works | `/how-it-works` | System explainer |
| Trade History | `/trade-history` | Dedicated page — KPI cards, win rate chart, filterable table |
| Settings | `/settings` | API keys, trading config, change password, reset trades |
| Admin | `/admin` | LLM costs, API usage, user management |
| Login | `/login` | Email + password, conditional 2FA |
| Register | `/register` | Email + password → 2FA QR setup → verify code |

### Onboarding Wizard (/onboarding)
- **Step 0**: Welcome — explains what Vesper does
- **Step 1**: Interest selection — Crypto, Stocks, Predictions (toggle cards)
- **Step 2**: Connect exchanges (optional) — Coinbase, Alpaca, Kalshi. Each expands inline form. All marked "Optional". Forms POST to settings endpoints with `redirect=/onboarding?step=2` to return after connecting.
- **Step 3**: Summary — shows paper balance, connection status. "Launch Dashboard" button POSTs to `/onboarding/complete`.

### Dashboard Tab (autopilot management)
- **Portfolio Overview** (top): Value, gradient chart (Total Value + Invested + Cash lines), stats grid
- **3 Autopilot Cards**: Crypto, Stocks, Predictions — each has ON/OFF state, fund/deployed/P&L display, single "Edit" button (opens unified modal with fund, max bet, risk, reinvest %)
- **Active Positions**: Cards with P&L sparkline, tooltips on all values, AI reasoning via `humanizeReason()`, Learn More modal (runs live deep research if no cached analysis)
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
- `openLearnMore(pid, name)` — opens Learn More modal, fetches `/api/position-analysis/{pid}`

## API Endpoints (Key Ones)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/portfolio-stats` | GET | Live portfolio value, P&L, win rate |
| `/api/positions` | GET | Open positions with live prices |
| `/api/altcoin-hunter` | GET/POST | Crypto autopilot status/control |
| `/api/autopilot` | GET/POST | Stocks autopilot status/control |
| `/api/predictions-autopilot` | GET/POST | Predictions autopilot status/control (now with real unrealized P&L) |
| `/api/position-analysis/{pid}` | GET | Full AI analysis — runs live Perplexity+Claude research if not cached |
| `/api/learning-state` | GET | Current learning engine state and strategy performance profiles |
| `/api/markets/crypto` | GET | Crypto market browser (60+ altcoins, live prices, market cap) |
| `/api/markets/stocks` | GET | Stock market browser (20 US equities, Yahoo Finance) |
| `/api/polymarket/markets` | GET | Polymarket prediction markets |
| `/api/kalshi/markets` | GET | Kalshi prediction markets |
| `/api/social-sentiment/{symbol}` | GET | X/Twitter social sentiment for a symbol |
| `/ws/live` | WS | Real-time updates (positions, stats, segments) every 2s |
| `/onboarding` | GET | Onboarding wizard page |
| `/onboarding/complete` | POST | Mark onboarding done, redirect to dashboard |
| `/settings/change-password` | POST | Change user password (complexity validated) |
| `/settings/reset-trades` | POST | Full portfolio reset — clears everything |
| `/settings/api-keys` | POST | Save/remove Coinbase keys (supports `redirect` param) |
| `/settings/alpaca-keys` | POST | Save/remove Alpaca keys (supports `redirect` param) |
| `/settings/kalshi-keys` | POST | Save/remove Kalshi keys (supports `redirect` param) |
| `/api/admin/bot-state` | GET | Debug: user configs, scan logs |
| `/api/admin/scan-test` | GET | Debug: test exchange + scanning pipeline |
| `/api/admin/deploy` | POST | Hot-deploy: git pull + restart |
| `/api/health` | GET | Health check with cycle state |

## Bot Cycle Flow (every 60 seconds)

```
UserBot.run_cycle():
  1. _process_symbol() for each watchlist + open position symbol
     → Fetches live market data (OHLCV, indicators)
     → Checks SL/TP/trailing stop for existing positions
     → Runs strategy analysis (BUY/SELL/HOLD signal)
     → Opens/closes positions based on signals

  2. _run_trend_scanner() — scans broader market for opportunities
  3. _run_autopilot() — AI-managed stock positions
  4. _run_altcoin_hunter() — 6-factor trend scoring on 60+ coins
     → Phase 1: BTC snapshot for relative strength
     → Phase 2: Parallel scan (ThreadPoolExecutor) of ALTCOIN_UNIVERSE
     → Phase 3: Rebalance — close positions with weakened trend scores
     → Phase 4: Deep research (Perplexity + Claude) on top 3 candidates
     → Phase 4b: Social sentiment boosting (X/Twitter)
     → Phase 5: Dynamic allocation — weight by score
     → Learning adjustments applied to: entry threshold, rebalance threshold,
       SL/TP percentages, position sizing

  5. _run_predictions_autopilot() — hourly scan of prediction markets
     → Fetches 30 Polymarket + 20 Kalshi markets
     → Filters by volume, skips held markets
     → Deep AI research on top 3 candidates (cost: ~$0.30-0.50/day)
     → Opens positions if AI edge >= min_edge_pct

  6. _refresh_prediction_prices() — updates prediction position values
     → Fetches current YES/NO prices from Polymarket + Kalshi
     → Updates current_probability in portfolio JSON
     → Checks SL/TP and auto-closes positions when hit
     → Adds prices to prices dict for snapshot recording

  7. _record_price_snapshots() — stores price history for sparkline charts
  8. _record_portfolio_snapshot() — stores portfolio value for 6-month chart
  9. _update_learning_state() — runs learning engine every 10 minutes
```

## Self-Learning Algorithm (vesper/learning.py)

### How It Works
1. **Performance Tracker**: Analyzes rolling window of last 50 trades per strategy
2. **Strategy Profiles**: Computes win rate, profit factor, avg win/loss, streaks, exit reason breakdown, trend (improving/declining/stable)
3. **Adaptive Adjustments**: Conservative parameter shifts with smoothing and hard caps
4. **Application**: Adjustments applied in `_run_altcoin_hunter()` to entry thresholds, SL/TP, position sizing, and rebalance thresholds

### Adjustment Logic
| Metric | Condition | Action |
|--------|-----------|--------|
| Win rate < 35% | Poor performance | Raise entry threshold +8% (be pickier) |
| Win rate < 45% | Below average | Raise entry threshold +4% |
| Win rate > 60% | Good performance | Lower entry threshold -3% (capture more) |
| Many SL exits + low WR | Cutting too late | Tighten stop-loss |
| Many SL exits + high WR | SL too tight | Loosen stop-loss |
| Avg win < avg loss | Bad R:R ratio | Widen take-profit |
| Profit factor > 1.5 | Strong returns | Increase position size (up to +15%) |
| Profit factor < 0.8 | Weak returns | Decrease position size (down to -20%) |
| Trend declining | Recent performance worse | Add urgency to threshold raise |

### Key Constants
- `MIN_TRADES_FOR_LEARNING = 8` (won't activate before 8 closed trades)
- `ROLLING_WINDOW = 50` (analyzes last 50 trades)
- `MAX_THRESHOLD_SHIFT = 0.15` (entry score can shift ±15%)
- `MAX_SL_TP_SHIFT = 0.30` (SL/TP can shift ±30%)
- Smoothing: moves 30% toward target each cycle
- Decay: adjustments shrink 5% per cycle when neutral

## Database Schema (SQLite)

### users table
```sql
id, email, password_hash, totp_secret,
coinbase_api_key, coinbase_api_secret,
alpaca_api_key, alpaca_api_secret,
kalshi_api_key, kalshi_api_secret,
perplexity_api_key,
paper_balance (default 500), trading_mode (paper/live),
symbols, stop_loss_pct, take_profit_min_pct, take_profit_max_pct,
max_position_pct, interval_minutes,
bot_active, is_admin, onboarding_complete,
created_at
```
All API keys are encrypted with Fernet (key stored at `data/.encryption_key`).

### trusted_devices table
```sql
id, user_id, token_hash (SHA-256), expires_at (7 days), created_at
```

### api_usage table
```sql
id, user_id, provider, model, input_tokens, output_tokens, cost_usd, endpoint, created_at
```

## Portfolio JSON Structure

Stored at `/app/data/portfolio_{user_id}.json`:
```json
{
  "cash": 495.0,
  "initial_balance": 500.0,
  "positions": {
    "BTC/USDT-1709000000": {
      "symbol", "side", "entry_price", "amount", "cost_usd",
      "entry_time", "strategy_reason", "id", "strategy_id",
      "bet_mode", "trade_mode", "stop_loss_pct", "tp_min_pct", "tp_max_pct",
      "trailing_stop_pct", "highest_price_seen",
      "limits": { "stop_loss_price", "take_profit_min_price", "take_profit_max_price", "position_size_usd", "position_size_asset", "trailing_stop_pct", "highest_price_seen" },
      // Extra fields (preserved by _PRESERVED_POS_KEYS):
      "price_history": [{"t": timestamp, "p": price, "w": win_prob}],
      "est_fee": 0.30,
      "current_probability": 0.65,  // predictions only
      "prediction_question", "prediction_side", "prediction_ai_prob", "prediction_mkt_prob", "prediction_edge"
    }
  },
  "trade_history": [
    { "symbol", "side", "entry_price", "exit_price", "amount",
      "pnl_usd", "pnl_pct", "entry_time", "exit_time", "reason",
      "strategy_reason", "trade_mode", "cost_usd",
      "entry_fee", "exit_fee", "total_fees", "net_pnl_usd", "net_pnl_pct" }
  ],
  "altcoin_hunter": { "enabled", "fund_usd", "max_positions", "risk_level", "trailing_stop_pct", "reinvest_pct", "max_bet_usd" },
  "autopilot": { "enabled", "fund_usd", "max_positions", "risk_level", "reinvest_pct", "max_bet_usd" },
  "predictions_autopilot": { "enabled", "fund_usd", "max_positions", "min_edge_pct", "risk_level", "reinvest_pct", "max_bet_usd", "last_scan_time" },
  "decision_log": [ { "action", "symbol", "strategy_id", "source", "trade_mode", "signal", "confidence", "reason", "time" } ],
  "autopilot_log": [ { "type", "markets_scanned", "status", "positions", "actions", "time" } ],
  "position_analyses": { "pid": { "deep_research": {...}, "indicators": {...}, "social_sentiment": {...} } },
  "portfolio_snapshots": [ { "t": timestamp, "v": value, "i": invested } ],
  "learning_state": {
    "status": "active",
    "total_trades_analyzed": 25,
    "strategy_profiles": { "altcoin_hunter": { "win_rate", "profit_factor", "trend", ... } },
    "adjustments": { "altcoin_hunter": { "entry_threshold_adj", "stop_loss_adj_pct", "take_profit_adj_pct", "position_size_multiplier", "rebalance_threshold_adj" } }
  }
}
```

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
| Prediction P&L stuck at $0 | `current_probability` never updated | Fixed — `_refresh_prediction_prices()` every cycle |
| Prediction unrealized P&L API = 0 | Hardcoded `"unrealized_pnl": 0` | Fixed — calculates from current_probability |
| Price_history lost after position changes | `_save_state()` didn't preserve extras | Fixed — `_PRESERVED_POS_KEYS` carries over fields |
| "No detailed analysis available" | Analysis not saved for all strategies | Fixed — runs live Perplexity+Claude research on-demand |
| Win rate declining | Static strategy parameters | Fixed — self-learning algorithm adjusts thresholds |

## Logging

Comprehensive logging via `_log = logging.getLogger("vesper.dashboard")` and `logging.getLogger("vesper.learning")`:
- Login attempts (success/fail/locked/2FA)
- Dashboard loads (user, mode, position count, trade count)
- API endpoint calls (portfolio-stats, positions, altcoin-hunter)
- WebSocket connections/disconnections
- Password changes, trade resets
- Learning engine updates (trades analyzed, adjustments applied)
- Prediction price refresh (Polymarket/Kalshi fetch results)
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

# Check learning state
curl -s http://localhost:8080/api/learning-state | python3 -m json.tool
```

## Current State (as of latest commit)

- **Crypto autopilot**: Working. BinanceUS auto-selected. Scans 60+ coins, opens positions. Learning engine adjusts entry thresholds and SL/TP based on performance.
- **Stocks autopilot**: Needs Alpaca API keys configured by user.
- **Predictions autopilot**: Works if Perplexity + Anthropic keys are set. Runs hourly scan. Prices refresh every cycle (~60s). SL/TP auto-close working. Real unrealized P&L shown in API.
- **Dashboard tab**: Real-time via WebSocket + 10s polling fallback. Portfolio gradient chart, tooltips, unified Edit modal, humanized AI decisions.
- **Markets tab**: 3 sub-tabs (Crypto, Stocks, Predictions). Crypto shows live prices for ALTCOIN_UNIVERSE + market cap via CoinGecko. Stocks via Yahoo Finance.
- **Trade History page**: Dedicated page with KPI cards per vertical, win rate time-series chart, filterable table (by type + date range).
- **Settings page**: Exchange connections, trading config, change password (complexity requirements), reset all trades (confirmation modal).
- **Onboarding wizard**: 4-step setup (Welcome → Interests → Exchange connections → Summary). Existing users auto-skip. New users redirected from /dashboard.
- **Login**: 2-step — hides 2FA field when device is trusted (7-day cookie).
- **Learn More modal**: Runs live deep research (Perplexity + Claude + social sentiment) on-demand if no cached analysis. Caches result for future requests.
- **Self-learning**: Adaptive strategy optimization every 10 minutes. Adjusts entry thresholds, SL/TP, position sizing based on trade history analysis.
- **Nav**: Consistent across all pages — Dashboard, Markets, How it works, Trade History, Settings, [Admin], Log out.
- **Logging**: Comprehensive — login, API calls, WebSocket, learning engine, prediction refresh, errors with tracebacks.
- **NoneType safety**: `_clean_nones()` at portfolio load time prevents all null-related crashes.
- **Price recording**: Every 60s per position (including predictions) for sparkline charts.
- **currentTradeMode**: Initialized from server-side `user.trading_mode` (not hardcoded 'paper').
- **Hot-deploy**: `/api/admin/deploy` does git pull + restart. Git installed in Dockerfile.

## Design System

### CSS Variables (base.html)
```css
--bg: #030306                    /* Near-black background */
--surface: rgba(255,255,255,0.025)
--card: rgba(255,255,255,0.035)
--border: rgba(255,255,255,0.06)
--text: #f0f0f5
--text-secondary: #9ca3af
--text-muted: #6b7280
--green: #00e85e                 /* Primary accent */
--green-dim: rgba(0,232,94,0.08)
--red: #ff4444
--accent: #00e85e
--glass: rgba(255,255,255,0.035)
--glass-border: rgba(255,255,255,0.07)
```

### Key CSS Classes
- `.card` — Glassmorphic card with backdrop blur, gradient background, inner glow
- `.btn-accent` — Green gradient button with shimmer hover effect
- `.btn-outline` — Glass button with blur
- `.nav` — Sticky glassmorphic header with blur + border glow
- `.nav-logo` — Shimmer gradient text animation
- `.input` — Glass input with accent focus ring
- `.toggle` — iOS-style toggle switch
- `.alert-success` / `.alert-error` — Green/red with dim background
- Font: Inter (Google Fonts), weights 300-800
