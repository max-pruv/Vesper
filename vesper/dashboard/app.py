"""Vesper Dashboard — Web UI for monitoring and managing the bot."""

import hashlib
import hmac
import os
import json
import secrets
from datetime import datetime

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Vesper Dashboard")

templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)

DATA_DIR = os.environ.get("VESPER_DATA_DIR", "data")
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "changeme123")

# Session tokens stored in memory (cleared on restart)
_valid_sessions: set[str] = set()


def _check_session(request: Request) -> bool:
    token = request.cookies.get("vesper_session")
    return token is not None and token in _valid_sessions


def _load_portfolio() -> dict:
    path = os.path.join(DATA_DIR, "portfolio.json")
    if not os.path.exists(path):
        return {
            "cash": 0,
            "initial_balance": 0,
            "positions": {},
            "trade_history": [],
        }
    with open(path) as f:
        return json.load(f)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = ""):
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": error,
    })


@app.post("/login")
async def login(request: Request, password: str = Form(...)):
    if hmac.compare_digest(password, DASHBOARD_PASSWORD):
        token = secrets.token_hex(32)
        _valid_sessions.add(token)
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(
            key="vesper_session",
            value=token,
            httponly=True,
            max_age=86400 * 7,  # 7 days
            samesite="strict",
        )
        return response
    return RedirectResponse(url="/login?error=wrong_password", status_code=303)


@app.get("/logout")
async def logout(request: Request):
    token = request.cookies.get("vesper_session")
    if token:
        _valid_sessions.discard(token)
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("vesper_session")
    return response


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if not _check_session(request):
        return RedirectResponse(url="/login", status_code=303)

    portfolio = _load_portfolio()

    # Compute stats
    cash = portfolio.get("cash", 0)
    initial = portfolio.get("initial_balance", 0)
    positions = portfolio.get("positions", {})
    trades = portfolio.get("trade_history", [])

    total_pnl = sum(t.get("pnl_usd", 0) for t in trades)
    wins = [t for t in trades if t.get("pnl_usd", 0) > 0]
    losses = [t for t in trades if t.get("pnl_usd", 0) <= 0]
    win_rate = (len(wins) / len(trades) * 100) if trades else 0

    # Equity curve from trade history
    equity_curve = [{"time": 0, "value": initial}]
    running = initial
    for t in trades:
        running += t.get("pnl_usd", 0)
        exit_time = t.get("exit_time", 0)
        equity_curve.append({
            "time": exit_time,
            "value": round(running, 2),
        })

    # Format trades for display
    formatted_trades = []
    for t in reversed(trades[-50:]):  # Last 50 trades, newest first
        entry_dt = datetime.fromtimestamp(t.get("entry_time", 0)).strftime("%m/%d %H:%M")
        exit_dt = datetime.fromtimestamp(t.get("exit_time", 0)).strftime("%m/%d %H:%M")
        formatted_trades.append({
            "symbol": t.get("symbol", ""),
            "side": t.get("side", ""),
            "entry_price": t.get("entry_price", 0),
            "exit_price": t.get("exit_price", 0),
            "amount": t.get("amount", 0),
            "pnl_usd": t.get("pnl_usd", 0),
            "pnl_pct": t.get("pnl_pct", 0),
            "entry_time": entry_dt,
            "exit_time": exit_dt,
            "reason": t.get("reason", ""),
        })

    # Format positions
    formatted_positions = []
    for pid, p in positions.items():
        entry_dt = datetime.fromtimestamp(p.get("entry_time", 0)).strftime("%m/%d %H:%M")
        formatted_positions.append({
            "id": pid,
            "symbol": p.get("symbol", ""),
            "side": p.get("side", ""),
            "entry_price": p.get("entry_price", 0),
            "amount": p.get("amount", 0),
            "cost_usd": p.get("cost_usd", 0),
            "entry_time": entry_dt,
            "stop_loss": p.get("limits", {}).get("stop_loss_price", 0),
            "tp_min": p.get("limits", {}).get("take_profit_min_price", 0),
            "tp_max": p.get("limits", {}).get("take_profit_max_price", 0),
        })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "cash": cash,
        "initial_balance": initial,
        "total_pnl": total_pnl,
        "total_pnl_pct": (total_pnl / initial * 100) if initial > 0 else 0,
        "win_rate": win_rate,
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "positions": formatted_positions,
        "trades": formatted_trades,
        "equity_curve": json.dumps(equity_curve),
        "mode": os.environ.get("TRADING_MODE", "paper").upper(),
    })


@app.get("/api/portfolio")
async def api_portfolio(request: Request):
    if not _check_session(request):
        return {"error": "unauthorized"}, 401
    return _load_portfolio()


@app.get("/api/health")
async def health():
    return {"status": "running", "time": datetime.now().isoformat()}
