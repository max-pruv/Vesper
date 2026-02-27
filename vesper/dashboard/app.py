"""Vesper Dashboard — SaaS platform with user accounts, 2FA, and API key management."""

import base64
import hmac
import io
import os
import json
import secrets
import time
from datetime import datetime

import bcrypt
import pyotp
import qrcode
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from vesper.dashboard.database import (
    init_db, create_user, get_user_by_email, get_user_by_id,
    verify_password, update_api_keys, update_trading_config,
    set_bot_active, User,
)

app = FastAPI(title="Vesper", docs_url=None, redoc_url=None)

templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)

DATA_DIR = os.environ.get("VESPER_DATA_DIR", "data")

# Sessions: token -> user_id
_sessions: dict[str, int] = {}

# Rate limiting
_failed_attempts: dict[str, list[float]] = {}
MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 300


@app.on_event("startup")
async def startup():
    init_db()


# --- Helpers ---

def _get_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _is_locked(ip: str) -> bool:
    now = time.time()
    attempts = [t for t in _failed_attempts.get(ip, []) if now - t < LOCKOUT_SECONDS]
    _failed_attempts[ip] = attempts
    return len(attempts) >= MAX_ATTEMPTS


def _record_fail(ip: str):
    _failed_attempts.setdefault(ip, []).append(time.time())


def _clear_fails(ip: str):
    _failed_attempts.pop(ip, None)


def _get_user(request: Request) -> User | None:
    token = request.cookies.get("vesper_session")
    if not token or token not in _sessions:
        return None
    return get_user_by_id(_sessions[token])


def _create_session(user_id: int) -> RedirectResponse:
    token = secrets.token_hex(32)
    _sessions[token] = user_id
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(
        key="vesper_session", value=token,
        httponly=True, max_age=86400 * 7, samesite="strict",
    )
    return response


def _load_user_portfolio(user_id: int) -> dict:
    path = os.path.join(DATA_DIR, f"portfolio_{user_id}.json")
    if not os.path.exists(path):
        return {"cash": 0, "initial_balance": 0, "positions": {}, "trade_history": []}
    with open(path) as f:
        return json.load(f)


# --- Home (login/register) ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    user = _get_user(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse("home.html", {"request": request})


# --- Register ---

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, error: str = ""):
    return templates.TemplateResponse("register.html", {"request": request, "error": error})


@app.post("/register")
async def register_step1(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(...),
):
    if len(password) < 8:
        return RedirectResponse(url="/register?error=password_short", status_code=303)
    if password != password_confirm:
        return RedirectResponse(url="/register?error=password_mismatch", status_code=303)

    existing = get_user_by_email(email)
    if existing:
        return RedirectResponse(url="/register?error=email_taken", status_code=303)

    # Generate TOTP secret and show QR setup
    totp_secret = pyotp.random_base32()
    totp = pyotp.TOTP(totp_secret)
    provisioning_uri = totp.provisioning_uri(name=email, issuer_name="Vesper")

    img = qrcode.make(provisioning_uri)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    qr_b64 = base64.b64encode(buffer.getvalue()).decode()

    # Hash password for the next step
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    return templates.TemplateResponse("register_2fa.html", {
        "request": request,
        "email": email,
        "pw_hash": pw_hash,
        "totp_secret": totp_secret,
        "qr_b64": qr_b64,
    })


@app.post("/register/verify-2fa")
async def register_verify_2fa(
    request: Request,
    email: str = Form(...),
    pw_hash: str = Form(...),
    totp_secret: str = Form(...),
    totp_code: str = Form(...),
):
    totp = pyotp.TOTP(totp_secret)
    if not totp.verify(totp_code, valid_window=1):
        return templates.TemplateResponse("register_2fa.html", {
            "request": request,
            "email": email,
            "pw_hash": pw_hash,
            "totp_secret": totp_secret,
            "qr_b64": "",  # Will re-generate
            "error": "invalid_code",
        })

    # Create user directly with pre-hashed password
    from vesper.dashboard.database import DB_PATH
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO users (email, password_hash, totp_secret, created_at) VALUES (?, ?, ?, ?)",
            (email.lower().strip(), pw_hash, totp_secret, time.time()),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return RedirectResponse(url="/register?error=email_taken", status_code=303)
    conn.close()

    user = get_user_by_email(email)
    if not user:
        return RedirectResponse(url="/register?error=unknown", status_code=303)

    return _create_session(user.id)


# --- Login ---

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = ""):
    ip = _get_ip(request)
    return templates.TemplateResponse("login.html", {
        "request": request, "error": error, "locked": _is_locked(ip),
    })


@app.post("/login")
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    totp_code: str = Form(""),
):
    ip = _get_ip(request)
    if _is_locked(ip):
        return RedirectResponse(url="/login?error=locked", status_code=303)

    user = get_user_by_email(email)
    if not user or not verify_password(user, password):
        _record_fail(ip)
        return RedirectResponse(url="/login?error=invalid", status_code=303)

    totp = pyotp.TOTP(user.totp_secret)
    if not totp.verify(totp_code, valid_window=1):
        _record_fail(ip)
        return RedirectResponse(url="/login?error=invalid_2fa", status_code=303)

    _clear_fails(ip)
    return _create_session(user.id)


# --- Logout ---

@app.get("/logout")
async def logout(request: Request):
    token = request.cookies.get("vesper_session")
    if token:
        _sessions.pop(token, None)
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie("vesper_session")
    return response


# --- Dashboard ---

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    portfolio = _load_user_portfolio(user.id)

    cash = portfolio.get("cash", 0)
    initial = portfolio.get("initial_balance", user.paper_balance)
    positions = portfolio.get("positions", {})
    trades = portfolio.get("trade_history", [])

    total_pnl = sum(t.get("pnl_usd", 0) for t in trades)
    wins = [t for t in trades if t.get("pnl_usd", 0) > 0]
    losses = [t for t in trades if t.get("pnl_usd", 0) <= 0]
    win_rate = (len(wins) / len(trades) * 100) if trades else 0

    equity_curve = [{"time": 0, "value": initial}]
    running = initial
    for t in trades:
        running += t.get("pnl_usd", 0)
        equity_curve.append({"time": t.get("exit_time", 0), "value": round(running, 2)})

    formatted_trades = []
    for t in reversed(trades[-50:]):
        formatted_trades.append({
            "symbol": t.get("symbol", ""),
            "side": t.get("side", ""),
            "entry_price": t.get("entry_price", 0),
            "exit_price": t.get("exit_price", 0),
            "pnl_usd": t.get("pnl_usd", 0),
            "pnl_pct": t.get("pnl_pct", 0),
            "entry_time": datetime.fromtimestamp(t.get("entry_time", 0)).strftime("%m/%d %H:%M"),
            "exit_time": datetime.fromtimestamp(t.get("exit_time", 0)).strftime("%m/%d %H:%M"),
            "reason": t.get("reason", ""),
        })

    formatted_positions = []
    for pid, p in positions.items():
        formatted_positions.append({
            "id": pid,
            "symbol": p.get("symbol", ""),
            "side": p.get("side", ""),
            "entry_price": p.get("entry_price", 0),
            "amount": p.get("amount", 0),
            "cost_usd": p.get("cost_usd", 0),
            "entry_time": datetime.fromtimestamp(p.get("entry_time", 0)).strftime("%m/%d %H:%M"),
            "stop_loss": p.get("limits", {}).get("stop_loss_price", 0),
            "tp_min": p.get("limits", {}).get("take_profit_min_price", 0),
            "tp_max": p.get("limits", {}).get("take_profit_max_price", 0),
        })

    has_api_keys = bool(user.coinbase_api_key)

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": user,
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
        "has_api_keys": has_api_keys,
    })


# --- Settings ---

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, msg: str = ""):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    return templates.TemplateResponse("settings.html", {
        "request": request,
        "user": user,
        "has_api_keys": bool(user.coinbase_api_key),
        "msg": msg,
    })


@app.post("/settings/api-keys")
async def save_api_keys(
    request: Request,
    api_key: str = Form(...),
    api_secret: str = Form(...),
):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    update_api_keys(user.id, api_key, api_secret)
    return RedirectResponse(url="/settings?msg=keys_saved", status_code=303)


@app.post("/settings/trading")
async def save_trading_config(
    request: Request,
    paper_balance: float = Form(500),
    trading_mode: str = Form("paper"),
    symbols: str = Form("BTC/USDT,ETH/USDT"),
    stop_loss_pct: float = Form(2.0),
    take_profit_min_pct: float = Form(1.5),
    take_profit_max_pct: float = Form(5.0),
    max_position_pct: float = Form(30.0),
    interval_minutes: int = Form(60),
):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    update_trading_config(
        user.id, paper_balance, trading_mode, symbols,
        stop_loss_pct, take_profit_min_pct, take_profit_max_pct,
        max_position_pct, interval_minutes,
    )
    return RedirectResponse(url="/settings?msg=config_saved", status_code=303)


@app.post("/settings/bot-toggle")
async def toggle_bot(request: Request):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    set_bot_active(user.id, not user.bot_active)
    return RedirectResponse(url="/dashboard", status_code=303)


# --- API ---

@app.get("/api/health")
async def health():
    return {"status": "running", "time": datetime.now().isoformat()}
