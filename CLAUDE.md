# Vesper — AI Trading Platform

## What is Vesper?
AI-powered crypto + stock trading bot with a SaaS dashboard. Users sign up, connect exchange API keys, and run an autopilot that scans markets, detects opportunities via a 9-layer AI brain, and executes trades automatically.

## Tech Stack
- **Backend**: Python 3.12, FastAPI, Uvicorn
- **Frontend**: Jinja2 templates, vanilla JS, Chart.js, TailwindCSS (dark theme)
- **Database**: SQLite (users, sessions) + JSON files (portfolios, trade history)
- **Deployment**: Docker Compose (Caddy reverse proxy + Vesper bot), Hostinger VPS
- **Exchanges**: Coinbase (crypto via ccxt), Alpaca (stocks)
- **CI/CD**: GitHub Actions auto-deploy on push to `main`

## Architecture

```
vesper/
├── main.py              # Bot engine: autopilot loop, signal processing, trade execution
├── market_data.py        # Technical indicators: RSI, MACD, Bollinger, S/R levels, volume profile, divergences
├── portfolio.py          # Portfolio management: positions, trades, value snapshots
├── risk.py               # Risk management: SL/TP, position sizing, trailing stops
├── exchange.py           # Exchange abstraction: Coinbase (ccxt) + Alpaca
├── humanize.py           # AI output humanizer: converts technical signals to plain English
├── whale_tracker.py      # Whale/large order detection
├── sentiment.py          # Market sentiment analysis
├── strategies/
│   ├── base.py           # Signal enum (BUY, SELL, SHORT, HOLD), StrategyResult
│   ├── ensemble.py       # EnhancedEnsemble: 9-layer AI brain (trend, momentum, whale, sentiment, ADX, RSI div, S/R, volume profile)
│   ├── trend_following.py
│   ├── mean_reversion.py
│   ├── momentum.py
│   └── catalog.py        # Strategy catalog for manual trading
├── dashboard/
│   ├── app.py            # FastAPI app: all API routes, WebSocket live updates, auth
│   ├── database.py       # SQLite: User model, sessions, CRUD
│   └── templates/
│       ├── base.html     # Base layout (nav, CSS, dark theme)
│       ├── home.html     # Landing page
│       ├── dashboard.html # Main dashboard: autopilot controls, positions, P&L chart
│       ├── markets.html   # Markets page: 3 tabs (Crypto, Stocks, Predictions)
│       ├── settings.html  # User settings
│       ├── login.html
│       └── register.html
config/
├── settings.py           # TICKER_SYMBOLS (20 crypto), STOCK_SYMBOLS (20 stocks), ALTCOIN_SCAN_SYMBOLS (20 altcoins), ALL_CRYPTO_SYMBOLS
```

## Deployment

### VPS Info
- **Host**: `srv1438426.hstgr.cloud` (Hostinger VPS, Ubuntu)
- **SSH**: Port 22 (currently blocked by Hostinger external firewall — use browser terminal)
- **Deploy key**: `~/.ssh/vesper_deploy_key` (passphrase-free, public key added to VPS authorized_keys)
- **GitHub secret**: `VPS_SSH_KEY` contains the deploy private key
- **Remote path**: `/root/vesper`

### Manual deploy via Hostinger browser terminal:
```bash
cd /root/vesper && git pull origin main && docker compose down && docker compose build --no-cache && docker compose up -d && docker compose logs --tail=10 vesper
```

### GitHub Actions auto-deploy:
Triggers on push to `main` via `.github/workflows/deploy.yml` (uses `appleboy/ssh-action`).
Currently blocked because SSH port 22 is not reachable externally from Hostinger's network firewall.

## Key Design Decisions

### 9-Layer EnhancedEnsemble AI Brain (vesper/strategies/ensemble.py)
1. Trend (EMA12/26, SMA50)
2. Momentum (RSI, MACD)
3. Whale activity
4. Sentiment (fear & greed)
5. Multi-timeframe confirmation
6. ADX trend strength
7. RSI Divergence (±0.10)
8. Support/Resistance proximity (±0.08)
9. Volume Profile POC positioning (±0.05)

### Signal types: BUY, SELL, SHORT, HOLD
- SHORT is distinct from SELL (SELL closes a long, SHORT opens a short position)
- Short selling supported via Alpaca (stocks) natively

### Portfolio value tracking
- `portfolio.py` records `value_history` snapshots each cycle
- Dashboard shows area chart (CoinMarketCap style) with gradient fill

### Humanizer (vesper/humanize.py)
- Converts technical AI signals to plain English narratives
- Handles BUY, SELL, SHORT signals with different narrative styles
- Uses `from __future__ import annotations` for Python 3.9 compatibility (local Mac)

### Price fetching
- Uses `t.get("last") or 0` pattern (not `t.get("last", 0)`) because exchanges can return `{"last": None}`
- price_map comprehensions use `(p["price"] or 0)` for safety

### User model (database.py)
- Has both Coinbase and Alpaca API key fields
- ALL functions that construct User() must include `alpaca_api_key` and `alpaca_api_secret`
- When reading from DB, use fallback: `row["alpaca_api_key"] if "alpaca_api_key" in row.keys() else ""`

## Common Pitfalls
- **Python 3.9 vs 3.12**: Local Mac is 3.9, Docker is 3.12. Use `from __future__ import annotations` for `dict | None` syntax.
- **Price None values**: Exchange APIs can return `None` for price fields. Always use `or 0` pattern.
- **User constructor**: Every place that creates a `User()` must include ALL fields including alpaca fields.
- **GitHub default branch**: Must be `main` (was accidentally set to a feature branch before).
- **Hostinger SSH**: Port 22 firewall rules are set but external access may be blocked at network level. Use browser terminal as fallback.

## Environment Variables (.env)
```
VESPER_DATA_DIR=data
DOMAIN=srv1438426.hstgr.cloud
GOOGLE_CLIENT_ID=        # For OAuth login
GOOGLE_CLIENT_SECRET=
APPLE_CLIENT_ID=         # For Apple login
```
API keys (Coinbase, Alpaca) are configured per-user via the dashboard settings page, not env vars.

## Current Status (March 2026)
- All 7 plan phases complete (P&L chart, positions fix, Markets page, AI humanizer, short selling, advanced indicators, altcoin scan)
- Production deployed on Hostinger VPS
- GitHub Actions CI/CD configured but SSH blocked externally — deploy via browser terminal for now
