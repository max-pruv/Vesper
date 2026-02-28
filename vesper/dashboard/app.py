"""Vesper Dashboard — SaaS with OAuth, 2FA, HTTPS."""

import asyncio
import base64
import hashlib
import hmac
import io
import os
import json
import secrets
import time
from datetime import datetime

import bcrypt
import ccxt
import httpx
import pyotp
import qrcode
from fastapi import FastAPI, Request, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from vesper.dashboard.database import (
    init_db, create_user, get_user_by_email, get_user_by_id,
    verify_password, update_api_keys, update_alpaca_keys, update_kalshi_keys,
    update_perplexity_key, update_trading_config,
    set_bot_active, update_trading_mode, create_oauth_user, User,
    add_trusted_device, is_device_trusted, remove_trusted_device,
    get_all_users, get_api_usage_summary, set_admin,
)
from vesper.strategies.catalog import get_strategy_catalog, get_strategy_by_id, STRATEGY_MAP as _SIGNAL_STRATEGY_MAP
from vesper.market_data import (
    get_market_snapshot, get_multi_tf_snapshot,
    get_order_book_pressure, fetch_fear_greed,
)
from vesper.portfolio import Portfolio, Position
from vesper.risk import PositionLimits

app = FastAPI(title="Vesper", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

DATA_DIR = os.environ.get("VESPER_DATA_DIR", "data")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
APPLE_CLIENT_ID = os.environ.get("APPLE_CLIENT_ID", "")
DOMAIN = os.environ.get("DOMAIN", "localhost")

_sessions: dict[str, int] = {}
_failed_attempts: dict[str, list[float]] = {}
# OAuth state tokens
_oauth_states: dict[str, float] = {}


@app.on_event("startup")
async def startup():
    init_db()


# --- Helpers ---

def _get_ip(r: Request) -> str:
    f = r.headers.get("x-forwarded-for")
    return f.split(",")[0].strip() if f else (r.client.host if r.client else "unknown")

def _is_locked(ip: str) -> bool:
    now = time.time()
    a = [t for t in _failed_attempts.get(ip, []) if now - t < 300]
    _failed_attempts[ip] = a
    return len(a) >= 5

def _record_fail(ip: str):
    _failed_attempts.setdefault(ip, []).append(time.time())

def _clear_fails(ip: str):
    _failed_attempts.pop(ip, None)

def _get_user(r: Request) -> User | None:
    t = r.cookies.get("vesper_session")
    return get_user_by_id(_sessions[t]) if t and t in _sessions else None

def _create_session(user_id: int) -> RedirectResponse:
    token = secrets.token_hex(32)
    _sessions[token] = user_id
    resp = RedirectResponse(url="/dashboard", status_code=303)
    resp.set_cookie(key="vesper_session", value=token, httponly=True,
                    max_age=86400 * 7, samesite="lax")
    return resp

def _oauth_context() -> dict:
    return {
        "google_enabled": bool(GOOGLE_CLIENT_ID),
        "apple_enabled": bool(APPLE_CLIENT_ID),
    }

def _load_portfolio(uid: int) -> dict:
    p = os.path.join(DATA_DIR, f"portfolio_{uid}.json")
    if not os.path.exists(p):
        return {"cash": 0, "initial_balance": 0, "positions": {}, "trade_history": []}
    with open(p) as f:
        return json.load(f)


def _save_portfolio(uid: int, data: dict):
    p = os.path.join(DATA_DIR, f"portfolio_{uid}.json")
    with open(p, "w") as f:
        json.dump(data, f, indent=2)


# --- Home ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if _get_user(request):
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse("home.html", {"request": request})


# --- Register ---

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, error: str = ""):
    return templates.TemplateResponse("register.html", {
        "request": request, "error": error, **_oauth_context(),
    })

@app.post("/register")
async def register_step1(request: Request, email: str = Form(...),
                         password: str = Form(...), password_confirm: str = Form(...)):
    if len(password) < 8:
        return RedirectResponse(url="/register?error=password_short", status_code=303)
    if password != password_confirm:
        return RedirectResponse(url="/register?error=password_mismatch", status_code=303)
    if get_user_by_email(email):
        return RedirectResponse(url="/register?error=email_taken", status_code=303)

    totp_secret = pyotp.random_base32()
    uri = pyotp.TOTP(totp_secret).provisioning_uri(name=email, issuer_name="Vesper")
    img = qrcode.make(uri)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    qr_b64 = base64.b64encode(buf.getvalue()).decode()
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    return templates.TemplateResponse("register_2fa.html", {
        "request": request, "email": email, "pw_hash": pw_hash,
        "totp_secret": totp_secret, "qr_b64": qr_b64,
    })

@app.post("/register/verify-2fa")
async def register_verify(request: Request, email: str = Form(...),
                          pw_hash: str = Form(...), totp_secret: str = Form(...),
                          totp_code: str = Form(...)):
    if not pyotp.TOTP(totp_secret).verify(totp_code, valid_window=1):
        return templates.TemplateResponse("register_2fa.html", {
            "request": request, "email": email, "pw_hash": pw_hash,
            "totp_secret": totp_secret, "qr_b64": "", "error": "invalid_code",
        })

    import sqlite3
    from vesper.dashboard.database import DB_PATH
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("INSERT INTO users (email, password_hash, totp_secret, created_at) VALUES (?,?,?,?)",
                     (email.lower().strip(), pw_hash, totp_secret, time.time()))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return RedirectResponse(url="/register?error=email_taken", status_code=303)
    conn.close()

    user = get_user_by_email(email)
    return _create_session(user.id) if user else RedirectResponse(url="/register", status_code=303)


# --- Login ---

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = ""):
    return templates.TemplateResponse("login.html", {
        "request": request, "error": error,
        "locked": _is_locked(_get_ip(request)), **_oauth_context(),
    })

@app.post("/login")
async def login(request: Request, email: str = Form(...),
                password: str = Form(...), totp_code: str = Form(""),
                remember_me: str = Form("")):
    ip = _get_ip(request)
    if _is_locked(ip):
        return RedirectResponse(url="/login?error=locked", status_code=303)
    user = get_user_by_email(email)
    if not user or not verify_password(user, password):
        _record_fail(ip)
        return RedirectResponse(url="/login?error=invalid", status_code=303)

    # Check if this device is trusted (skip 2FA)
    trust_cookie = request.cookies.get("vesper_trust", "")
    device_trusted = False
    if trust_cookie:
        token_hash = hashlib.sha256(trust_cookie.encode()).hexdigest()
        device_trusted = is_device_trusted(user.id, token_hash)

    if not device_trusted:
        if not pyotp.TOTP(user.totp_secret).verify(totp_code, valid_window=1):
            _record_fail(ip)
            return RedirectResponse(url="/login?error=invalid_2fa", status_code=303)

    _clear_fails(ip)
    resp = _create_session(user.id)

    # Set trust cookie if "remember me" checked
    if remember_me:
        trust_token = secrets.token_hex(32)
        token_hash = hashlib.sha256(trust_token.encode()).hexdigest()
        expires_at = time.time() + 86400 * 7  # 7 days
        add_trusted_device(user.id, token_hash, expires_at)
        resp.set_cookie(key="vesper_trust", value=trust_token, httponly=True,
                        max_age=86400 * 7, samesite="lax")

    return resp


# --- Google OAuth ---

@app.get("/auth/google")
async def google_auth(request: Request):
    if not GOOGLE_CLIENT_ID:
        return RedirectResponse(url="/login", status_code=303)
    state = secrets.token_hex(16)
    _oauth_states[state] = time.time()
    scheme = "https" if DOMAIN != "localhost" else "http"
    port_part = ":8080" if DOMAIN == "localhost" else ""
    redirect_uri = f"{scheme}://{DOMAIN}{port_part}/auth/google/callback"
    url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={GOOGLE_CLIENT_ID}&response_type=code&"
        f"scope=email%20profile&state={state}&"
        f"redirect_uri={redirect_uri}"
    )
    return RedirectResponse(url=url)

@app.get("/auth/google/callback")
async def google_callback(request: Request, code: str = "", state: str = ""):
    if state not in _oauth_states or time.time() - _oauth_states.pop(state) > 600:
        return RedirectResponse(url="/login?error=invalid", status_code=303)

    scheme = "https" if DOMAIN != "localhost" else "http"
    port_part = ":8080" if DOMAIN == "localhost" else ""
    redirect_uri = f"{scheme}://{DOMAIN}{port_part}/auth/google/callback"

    async with httpx.AsyncClient() as client:
        token_resp = await client.post("https://oauth2.googleapis.com/token", data={
            "code": code, "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": redirect_uri, "grant_type": "authorization_code",
        })
        if token_resp.status_code != 200:
            return RedirectResponse(url="/login?error=invalid", status_code=303)

        tokens = token_resp.json()
        user_resp = await client.get("https://www.googleapis.com/oauth2/v2/userinfo",
                                     headers={"Authorization": f"Bearer {tokens['access_token']}"})
        if user_resp.status_code != 200:
            return RedirectResponse(url="/login?error=invalid", status_code=303)

        info = user_resp.json()
        email = info.get("email", "").lower().strip()
        if not email:
            return RedirectResponse(url="/login?error=invalid", status_code=303)

    user = get_user_by_email(email)
    if not user:
        user = create_oauth_user(email)
    if not user:
        return RedirectResponse(url="/login?error=invalid", status_code=303)
    return _create_session(user.id)


# --- Apple OAuth ---

@app.get("/auth/apple")
async def apple_auth(request: Request):
    if not APPLE_CLIENT_ID:
        return RedirectResponse(url="/login", status_code=303)
    state = secrets.token_hex(16)
    _oauth_states[state] = time.time()
    scheme = "https" if DOMAIN != "localhost" else "http"
    port_part = ":8080" if DOMAIN == "localhost" else ""
    redirect_uri = f"{scheme}://{DOMAIN}{port_part}/auth/apple/callback"
    url = (
        f"https://appleid.apple.com/auth/authorize?"
        f"client_id={APPLE_CLIENT_ID}&response_type=code%20id_token&"
        f"scope=email%20name&response_mode=form_post&state={state}&"
        f"redirect_uri={redirect_uri}"
    )
    return RedirectResponse(url=url)

@app.post("/auth/apple/callback")
async def apple_callback(request: Request):
    form = await request.form()
    state = form.get("state", "")
    id_token = form.get("id_token", "")

    if state not in _oauth_states or time.time() - _oauth_states.pop(state) > 600:
        return RedirectResponse(url="/login?error=invalid", status_code=303)

    # Decode JWT payload (Apple id_token is a JWT)
    try:
        payload = id_token.split(".")[1]
        payload += "=" * (4 - len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload))
        email = data.get("email", "").lower().strip()
    except Exception:
        return RedirectResponse(url="/login?error=invalid", status_code=303)

    if not email:
        return RedirectResponse(url="/login?error=invalid", status_code=303)

    user = get_user_by_email(email)
    if not user:
        user = create_oauth_user(email)
    if not user:
        return RedirectResponse(url="/login?error=invalid", status_code=303)
    return _create_session(user.id)


# --- Logout ---

@app.get("/logout")
async def logout(request: Request):
    t = request.cookies.get("vesper_session")
    if t:
        _sessions.pop(t, None)
    # Remove trusted device token
    trust_cookie = request.cookies.get("vesper_trust", "")
    if trust_cookie:
        token_hash = hashlib.sha256(trust_cookie.encode()).hexdigest()
        remove_trusted_device(token_hash)
    resp = RedirectResponse(url="/", status_code=303)
    resp.delete_cookie("vesper_session")
    resp.delete_cookie("vesper_trust")
    return resp


# --- How It Works ---

@app.get("/how-it-works", response_class=HTMLResponse)
async def how_it_works(request: Request):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("how_it_works.html", {"request": request})


# --- Dashboard ---

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    portfolio = _load_portfolio(user.id)
    cash = portfolio.get("cash", user.paper_balance)
    initial = portfolio.get("initial_balance", user.paper_balance)
    all_positions = portfolio.get("positions", {})
    all_trades = portfolio.get("trade_history", [])

    # Silo by trade_mode: only show positions/trades matching the user's active mode
    active_mode = user.trading_mode  # "paper" or "live"
    mode_match = {"live": "real", "paper": "paper"}
    current_mode = mode_match.get(active_mode, "paper")
    positions = {
        pid: p for pid, p in all_positions.items()
        if p.get("trade_mode", "paper") == current_mode
    }
    trades = [
        t for t in all_trades
        if t.get("trade_mode", "paper") == current_mode
    ]
    realized_pnl = sum(t.get("pnl_usd", 0) for t in trades)
    total_pnl = realized_pnl  # unrealized added by WS once live prices arrive
    wins = [t for t in trades if t.get("pnl_usd", 0) > 0]
    losses = [t for t in trades if t.get("pnl_usd", 0) <= 0]
    win_rate = (len(wins) / len(trades) * 100) if trades else 0

    equity = [{"time": 0, "value": initial}]
    running = initial
    for t in trades:
        running += t.get("pnl_usd", 0)
        equity.append({"time": t.get("exit_time", 0), "value": round(running, 2)})

    fmt_trades = []
    for t in reversed(trades[-50:]):
        fmt_trades.append({
            "symbol": t.get("symbol", ""), "side": t.get("side", ""),
            "entry_price": t.get("entry_price", 0), "exit_price": t.get("exit_price", 0),
            "pnl_usd": t.get("pnl_usd", 0), "pnl_pct": t.get("pnl_pct", 0),
            "entry_time": datetime.fromtimestamp(t.get("entry_time", 0)).strftime("%m/%d %H:%M"),
            "exit_time": datetime.fromtimestamp(t.get("exit_time", 0)).strftime("%m/%d %H:%M"),
            "reason": t.get("reason", ""),
        })

    fmt_pos = []
    for pid, p in positions.items():
        entry = p.get("entry_price", 0)
        amount = p.get("amount", 0)
        side = p.get("side", "buy")
        sl_price = p.get("limits", {}).get("stop_loss_price", 0)
        tp_max_price = p.get("limits", {}).get("take_profit_max_price", 0)
        cost = p.get("cost_usd", 0)
        if side == "buy":
            ml = (entry - sl_price) * amount if sl_price > 0 else cost
            mw = (tp_max_price - entry) * amount if tp_max_price > 0 else 0
        else:
            ml = (sl_price - entry) * amount if sl_price > 0 else cost
            mw = (entry - tp_max_price) * amount if tp_max_price > 0 else 0
        fmt_pos.append({
            "id": pid, "symbol": p.get("symbol", ""), "side": side,
            "entry_price": entry, "amount": amount,
            "cost_usd": cost,
            "entry_time": datetime.fromtimestamp(p.get("entry_time", 0)).strftime("%m/%d %H:%M"),
            "stop_loss": sl_price,
            "tp_min": p.get("limits", {}).get("take_profit_min_price", 0),
            "tp_max": tp_max_price,
            "max_loss": round(abs(ml), 2),
            "max_win": round(abs(mw), 2),
            "bet_mode": p.get("bet_mode", "one_off"),
            "trade_mode": p.get("trade_mode", "paper"),
            "est_fee": p.get("est_fee", round(cost * 0.006, 2)),
        })

    strategies = get_strategy_catalog()

    return templates.TemplateResponse("dashboard.html", {
        "request": request, "user": user, "cash": cash,
        "initial_balance": initial, "total_pnl": total_pnl,
        "realized_pnl": realized_pnl,
        "total_pnl_pct": (total_pnl / initial * 100) if initial > 0 else 0,
        "win_rate": win_rate, "total_trades": len(trades),
        "wins": len(wins), "losses": len(losses),
        "positions": fmt_pos, "trades": fmt_trades,
        "equity_curve": json.dumps(equity),
        "has_api_keys": bool(user.coinbase_api_key) or bool(user.alpaca_api_key),
        "has_alpaca": bool(user.alpaca_api_key),
        "strategies": strategies,
    })


# --- Settings ---

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, msg: str = ""):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("settings.html", {
        "request": request, "user": user,
        "has_api_keys": bool(user.coinbase_api_key),
        "has_coinbase": bool(user.coinbase_api_key),
        "has_alpaca": bool(user.alpaca_api_key),
        "has_kalshi": bool(user.kalshi_api_key),
        "has_perplexity": bool(user.perplexity_api_key),
        "msg": msg,
    })

@app.post("/settings/api-keys")
async def save_keys(request: Request, api_key: str = Form(...), api_secret: str = Form(...)):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    update_api_keys(user.id, api_key, api_secret)
    return RedirectResponse(url="/settings?msg=keys_saved", status_code=303)

@app.post("/settings/api-keys/remove")
async def remove_keys(request: Request):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    update_api_keys(user.id, "", "")
    return RedirectResponse(url="/settings?msg=keys_saved", status_code=303)

@app.post("/settings/alpaca-keys")
async def save_alpaca_keys(request: Request, alpaca_key: str = Form(...), alpaca_secret: str = Form(...)):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    update_alpaca_keys(user.id, alpaca_key, alpaca_secret)
    return RedirectResponse(url="/settings?msg=keys_saved", status_code=303)

@app.post("/settings/alpaca-keys/remove")
async def remove_alpaca_keys(request: Request):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    update_alpaca_keys(user.id, "", "")
    return RedirectResponse(url="/settings?msg=keys_saved", status_code=303)

@app.post("/settings/kalshi-keys")
async def save_kalshi_keys(request: Request, kalshi_key: str = Form(...), kalshi_secret: str = Form(...)):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    update_kalshi_keys(user.id, kalshi_key, kalshi_secret)
    return RedirectResponse(url="/settings?msg=keys_saved", status_code=303)

@app.post("/settings/kalshi-keys/remove")
async def remove_kalshi_keys(request: Request):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    update_kalshi_keys(user.id, "", "")
    return RedirectResponse(url="/settings?msg=keys_saved", status_code=303)

@app.post("/settings/perplexity-key")
async def save_perplexity_key(request: Request, perplexity_key: str = Form(...)):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    update_perplexity_key(user.id, perplexity_key)
    return RedirectResponse(url="/settings?msg=keys_saved", status_code=303)

@app.post("/settings/perplexity-key/remove")
async def remove_perplexity_key(request: Request):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    update_perplexity_key(user.id, "")
    return RedirectResponse(url="/settings?msg=keys_saved", status_code=303)

@app.post("/settings/trading")
async def save_config(request: Request, paper_balance: float = Form(500),
                      trading_mode: str = Form("paper"),
                      symbols: str = Form("BTC/USDT,ETH/USDT"),
                      stop_loss_pct: float = Form(2.0),
                      take_profit_min_pct: float = Form(1.5),
                      take_profit_max_pct: float = Form(5.0),
                      max_position_pct: float = Form(30.0),
                      interval_minutes: int = Form(60)):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    update_trading_config(user.id, paper_balance, trading_mode, symbols,
                          stop_loss_pct, take_profit_min_pct, take_profit_max_pct,
                          max_position_pct, interval_minutes)
    return RedirectResponse(url="/settings?msg=config_saved", status_code=303)

@app.post("/settings/bot-toggle")
async def toggle_bot(request: Request):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    set_bot_active(user.id, not user.bot_active)
    return RedirectResponse(url="/dashboard", status_code=303)

@app.post("/settings/trading-mode")
async def toggle_mode(request: Request):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    new_mode = "live" if user.trading_mode == "paper" else "paper"
    update_trading_mode(user.id, new_mode)
    return RedirectResponse(url="/dashboard", status_code=303)


# --- Admin ---

ADMIN_TOKEN = os.environ.get("VESPER_ADMIN_TOKEN", "")


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, days: int = 30):
    """Admin dashboard — LLM costs, users, API usage. Admin-only."""
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    # Bootstrap: if no admin exists yet, promote the current user
    all_users = get_all_users()
    has_any_admin = any(u.is_admin for u in all_users)
    if not has_any_admin and all_users:
        set_admin(user.id, True)
        user = get_user_by_id(user.id)

    if not user.is_admin:
        return RedirectResponse(url="/dashboard", status_code=303)

    days = max(1, min(365, days))
    usage = get_api_usage_summary(days=days)

    # Format recent calls timestamps
    for r in usage["recent"]:
        r["time_fmt"] = datetime.fromtimestamp(r["created_at"]).strftime("%m/%d %H:%M")

    # Format user join dates
    user_list = []
    for u in all_users:
        user_list.append({
            "id": u.id,
            "email": u.email,
            "trading_mode": u.trading_mode,
            "bot_active": u.bot_active,
            "is_admin": u.is_admin,
            "has_coinbase": u.has_coinbase,
            "has_alpaca": u.has_alpaca,
            "has_kalshi": u.has_kalshi,
            "joined": datetime.fromtimestamp(u.created_at).strftime("%Y-%m-%d"),
        })

    # Count actual active autopilots across all users
    active_bots = 0
    for u in all_users:
        p = _load_portfolio(u.id)
        if p.get("altcoin_hunter", {}).get("enabled"):
            active_bots += 1
        if p.get("autopilot", {}).get("enabled"):
            active_bots += 1
        if p.get("predictions_autopilot", {}).get("enabled"):
            active_bots += 1

    # Check if LLM API keys are configured
    import os as _os
    ai_keys = {
        "perplexity": bool(_os.environ.get("PERPLEXITY_API_KEY", "")),
        "anthropic": bool(_os.environ.get("ANTHROPIC_API_KEY", "")),
    }

    return templates.TemplateResponse("admin.html", {
        "request": request,
        "user": user,
        "users": user_list,
        "usage": usage,
        "days": days,
        "active_bots": active_bots,
        "ai_keys": ai_keys,
    })


@app.post("/admin/promote")
async def admin_promote(request: Request):
    """Promote a user to admin by email (admin-only)."""
    user = _get_user(request)
    if not user or not user.is_admin:
        return JSONResponse({"error": "forbidden"}, status_code=403)
    body = await request.json()
    target_email = body.get("email", "").strip()
    target = get_user_by_email(target_email)
    if not target:
        return JSONResponse({"error": "User not found"}, status_code=404)
    set_admin(target.id, True)
    return {"ok": True}


# --- Price Cache ---

from config.settings import TICKER_SYMBOLS
_price_cache: dict = {}
_price_cache_time: float = 0
_CACHE_TTL = 10  # seconds

# Signal cache: {symbol: {"data": {...}, "time": float}}
_signal_cache: dict = {}
_SIGNAL_CACHE_TTL = 30  # seconds


def _fetch_signal_sync(symbol: str, strategy_id: str = "smart_auto") -> dict:
    """Run per-strategy analysis on a symbol (sync, for run_in_executor)."""
    try:
        config = _SIGNAL_STRATEGY_MAP.get(strategy_id)
        if not config or config["strategy"] is None:
            return {"signal": "HOLD", "confidence": 0,
                    "reason": "Set & Forget — no AI analysis, monitors SL/TP only",
                    "strategy": strategy_id}

        ex = _get_public_exchange()
        timeframe = config["timeframe"]

        if timeframe == "multi":
            snapshot = get_multi_tf_snapshot(ex, symbol)
            # Enrich with order book + sentiment
            try:
                ob = get_order_book_pressure(ex, symbol)
                snapshot["buy_pressure"] = ob["buy_pressure"]
                snapshot["spread_pct"] = ob["spread_pct"]
            except Exception:
                snapshot["buy_pressure"] = 0.5
            try:
                snapshot["fear_greed"] = fetch_fear_greed()
            except Exception:
                snapshot["fear_greed"] = 50
        else:
            snapshot = get_market_snapshot(ex, symbol, timeframe=timeframe)

        strategy = config["strategy"]()
        result = strategy.analyze(snapshot)
        return {
            "signal": result.signal.value,
            "confidence": round(result.confidence, 2),
            "reason": result.reason,
            "strategy": strategy_id,
            "timeframe": timeframe if timeframe != "multi" else "1h+4h",
        }
    except Exception:
        return {"signal": "HOLD", "confidence": 0,
                "reason": "Unable to analyze — data unavailable",
                "strategy": strategy_id}


def _get_public_exchange():
    """Public (no auth) exchange for price data. Uses Binance for reliable USDT pairs."""
    return ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})


def _calc_change_pct(t: dict) -> float:
    """Calculate 24h change % from ticker data."""
    pct = t.get("percentage")
    if pct is not None and pct != 0:
        return round(pct, 2)
    last = t.get("last", 0)
    opening = t.get("open")
    if opening and opening > 0 and last:
        return round(((last - opening) / opening) * 100, 2)
    return 0.0


def _fetch_tickers_sync() -> list[dict]:
    """Fetch ticker data for all symbols (sync, run in executor)."""
    ex = _get_public_exchange()
    result = []
    try:
        tickers = ex.fetch_tickers(TICKER_SYMBOLS)
        for sym in TICKER_SYMBOLS:
            t = tickers.get(sym)
            if t:
                result.append({
                    "symbol": sym,
                    "name": sym.split("/")[0],
                    "price": t.get("last", 0),
                    "change_pct": _calc_change_pct(t),
                })
    except Exception:
        # Fallback: fetch individually
        for sym in TICKER_SYMBOLS:
            try:
                t = ex.fetch_ticker(sym)
                result.append({
                    "symbol": sym,
                    "name": sym.split("/")[0],
                    "price": t.get("last", 0),
                    "change_pct": _calc_change_pct(t),
                })
            except Exception:
                pass
    return result


def _fetch_ohlcv_sync(symbol: str, timeframe: str = "1h", limit: int = 100) -> list[dict]:
    """Fetch OHLCV candles formatted for lightweight-charts."""
    ex = _get_public_exchange()
    try:
        raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return [
            {"time": int(c[0] / 1000), "open": c[1], "high": c[2], "low": c[3], "close": c[4]}
            for c in raw
        ]
    except Exception:
        return []


async def _get_cached_prices() -> list[dict]:
    """Return cached ticker prices, refreshing if stale."""
    global _price_cache, _price_cache_time
    now = time.time()
    if _price_cache and (now - _price_cache_time) < _CACHE_TTL:
        return _price_cache
    loop = asyncio.get_event_loop()
    _price_cache = await loop.run_in_executor(None, _fetch_tickers_sync)
    _price_cache_time = now
    return _price_cache


def _fetch_single_price(symbol: str) -> float:
    """Fetch a single crypto symbol's price via Binance."""
    try:
        ex = _get_public_exchange()
        t = ex.fetch_ticker(symbol)
        return t.get("last", 0) if t else 0
    except Exception:
        return 0


def _fetch_stock_price(symbol: str, user) -> float:
    """Fetch a stock price via Alpaca (requires user with Alpaca keys)."""
    if not user or not user.has_alpaca:
        return 0
    try:
        from vesper.exchange import AlpacaExchange
        alpaca = AlpacaExchange(
            api_key=user.get_alpaca_key(),
            api_secret=user.get_alpaca_secret(),
            paper=False,
        )
        t = alpaca.fetch_ticker(symbol)
        return t.get("last", 0) if t else 0
    except Exception:
        return 0


async def _get_price_map_for_positions(position_symbols: set[str], user=None) -> dict[str, float]:
    """Build a complete price map covering all position symbols."""
    prices = await _get_cached_prices()
    price_map = {p["symbol"]: p["price"] for p in prices}

    # Fetch missing symbols individually
    missing = position_symbols - set(price_map.keys())
    if missing:
        loop = asyncio.get_event_loop()
        for sym in missing:
            if sym.endswith("/USD"):
                # Stock symbol — use Alpaca
                price = await loop.run_in_executor(None, _fetch_stock_price, sym, user)
            else:
                # Crypto symbol — use Binance
                price = await loop.run_in_executor(None, _fetch_single_price, sym)
            if price > 0:
                price_map[sym] = price

    return price_map


def _build_segment_statuses(portfolio: dict, pos_list: list) -> dict:
    """Build bot segment statuses for WS payload."""
    segments = {}
    strategy_map = {
        "crypto": ("altcoin_hunter", "altcoin_hunter"),
        "stocks": ("autopilot", "autopilot"),
        "predictions": ("predictions_autopilot", "predictions"),
    }
    all_logs = portfolio.get("autopilot_log", [])
    all_trades = portfolio.get("trade_history", [])

    for seg_name, (cfg_key, strategy_id) in strategy_map.items():
        cfg = portfolio.get(cfg_key, {})
        seg_positions = [p for p in pos_list if p.get("strategy_id") == strategy_id]
        deployed = sum(p.get("cost_usd", 0) for p in seg_positions)
        unrealized = sum(p.get("pnl_usd", 0) for p in seg_positions)
        realized = sum(
            t.get("pnl_usd", 0) for t in all_trades
            if t.get("strategy_id") == strategy_id
        )
        fund = cfg.get("fund_usd", 0)

        # Last scan info
        log_type = "altcoin_hunter_scan" if seg_name == "crypto" else (
            "predictions_scan" if seg_name == "predictions" else "autopilot_scan"
        )
        seg_logs = [l for l in all_logs if l.get("type") == log_type][-5:]

        segments[seg_name] = {
            "enabled": cfg.get("enabled", False),
            "fund_usd": fund,
            "available_usd": round(fund - deployed, 2),
            "deployed_usd": round(deployed, 2),
            "unrealized_pnl": round(unrealized, 2),
            "realized_pnl": round(realized, 2),
            "positions_count": len(seg_positions),
            "max_bet_usd": cfg.get("max_bet_usd", 0),
            "risk_level": cfg.get("risk_level", "aggressive"),
            "reinvest_pct": cfg.get("reinvest_pct", 100),
            "log": seg_logs,
        }
    return segments


# --- API Endpoints ---

@app.get("/api/health")
async def health():
    import os as _os
    return {
        "status": "running",
        "time": datetime.now().isoformat(),
        "ai_keys": {
            "perplexity": bool(_os.environ.get("PERPLEXITY_API_KEY", "")),
            "anthropic": bool(_os.environ.get("ANTHROPIC_API_KEY", "")),
        },
    }


@app.get("/api/admin/diagnostics")
async def admin_diagnostics(request: Request):
    """Debug endpoint — check API usage DB, test LLM keys."""
    user = _get_user(request)
    if not user or not user.is_admin:
        return {"error": "admin only"}

    import os as _os
    import sqlite3

    # Check api_usage table
    db_path = os.path.join(
        _os.environ.get("VESPER_DATA_DIR", "data"), "vesper.db"
    )
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        row_count = conn.execute("SELECT COUNT(*) as n FROM api_usage").fetchone()["n"]
        recent = [dict(r) for r in conn.execute(
            "SELECT provider, model, cost_usd, endpoint, created_at FROM api_usage ORDER BY created_at DESC LIMIT 5"
        ).fetchall()]
        conn.close()
    except Exception as e:
        row_count = -1
        recent = [{"error": str(e)}]

    # Quick test: Perplexity key validity (lightweight)
    pplx_ok = None
    pplx_key = _os.environ.get("PERPLEXITY_API_KEY", "")
    if pplx_key:
        try:
            import httpx
            r = httpx.post(
                "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {pplx_key}", "Content-Type": "application/json"},
                json={"model": "sonar", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5},
                timeout=10,
            )
            pplx_ok = r.status_code == 200
            if not pplx_ok:
                pplx_ok = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            pplx_ok = f"error: {e}"

    anth_ok = None
    anth_key = _os.environ.get("ANTHROPIC_API_KEY", "")
    if anth_key:
        try:
            import httpx
            r = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": anth_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                json={"model": "claude-haiku-4-5-20251001", "max_tokens": 5, "messages": [{"role": "user", "content": "ping"}]},
                timeout=10,
            )
            anth_ok = r.status_code == 200
            if not anth_ok:
                anth_ok = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            anth_ok = f"error: {e}"

    return {
        "db_path": db_path,
        "api_usage_rows": row_count,
        "recent_usage": recent,
        "perplexity_test": pplx_ok,
        "anthropic_test": anth_ok,
    }


# ── WebSocket: real-time position & portfolio updates ──

_ws_connections: set[WebSocket] = set()


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    """WebSocket for real-time position and portfolio updates.

    Client sends: {"type": "auth", "session": "<cookie_value>"}
    Server pushes: positions + portfolio stats every 2 seconds.
    """
    await ws.accept()
    _ws_connections.add(ws)
    user = None

    try:
        # Wait for auth message
        auth_msg = await asyncio.wait_for(ws.receive_json(), timeout=5)
        session_id = auth_msg.get("session", "")
        user_id = _sessions.get(session_id)
        if not user_id:
            await ws.send_json({"type": "error", "msg": "unauthorized"})
            return
        user = get_user_by_id(user_id)
        if not user:
            await ws.send_json({"type": "error", "msg": "unauthorized"})
            return

        await ws.send_json({"type": "auth_ok"})

        # Push loop: send fresh data every 2 seconds
        while True:
            try:
                portfolio = _load_portfolio(user.id)
                all_positions = portfolio.get("positions", {})

                # --- Positions (send ALL, client filters by trade_mode) ---
                positions = all_positions
                pos_list = []
                if positions:
                    pos_symbols = {
                        p.get("symbol", "") for p in positions.values()
                        if not p.get("symbol", "").startswith("PRED:")
                    }
                    price_map = await _get_price_map_for_positions(pos_symbols, user)

                    for pid, p in positions.items():
                        sym = p.get("symbol", "")
                        entry = p.get("entry_price", 0)
                        if sym.startswith("PRED:"):
                            current = p.get("current_probability", entry)
                        else:
                            current = price_map.get(sym, entry)
                        amount = p.get("amount", 0)
                        side = p.get("side", "buy")
                        pnl_usd = (current - entry) * amount if side == "buy" else (entry - current) * amount
                        pnl_pct = ((current - entry) / entry * 100) if entry > 0 and side == "buy" else (
                            ((entry - current) / entry * 100) if entry > 0 else 0)

                        limits = p.get("limits", {})
                        sl = limits.get("stop_loss_price", 0)
                        tp_max = limits.get("take_profit_max_price", 0)
                        price_range = tp_max - sl if tp_max > sl else 1
                        progress = max(0, min(100, ((current - sl) / price_range) * 100))

                        cost_usd = p.get("cost_usd", 0)
                        tp_min = limits.get("take_profit_min_price", 0)
                        if side == "buy":
                            max_loss = abs((entry - sl) * amount) if sl > 0 else cost_usd
                            max_win = abs((tp_max - entry) * amount) if tp_max > 0 else 0
                        else:
                            max_loss = abs((sl - entry) * amount) if sl > 0 else cost_usd
                            max_win = abs((entry - tp_max) * amount) if tp_max > 0 else 0

                        trailing_pct = p.get("trailing_stop_pct", 0)
                        highest_seen = p.get("highest_price_seen", 0)
                        trailing_sl = highest_seen * (1 - trailing_pct / 100) if trailing_pct > 0 and highest_seen > 0 else 0

                        price_snapshots = p.get("price_history", [])
                        now_ts = time.time()
                        if len(price_snapshots) == 0 or (now_ts - (price_snapshots[-1].get("t", 0))) > 30:
                            effective_sl = max(sl, trailing_sl) if trailing_sl > 0 else sl
                            range_total = tp_max - effective_sl if tp_max > effective_sl else 1
                            win_prob = max(0, min(100, ((current - effective_sl) / range_total) * 100))
                            price_snapshots.append({"t": now_ts, "p": current, "w": round(win_prob, 1)})
                            if len(price_snapshots) > 60:
                                price_snapshots = price_snapshots[-60:]
                            p["price_history"] = price_snapshots
                            _save_portfolio(user.id, portfolio)

                        pos_list.append({
                            "id": pid, "symbol": sym,
                            "name": sym.split("/")[0] if sym else "",
                            "side": side, "entry_price": entry,
                            "current_price": current, "cost_usd": cost_usd,
                            "amount": amount,
                            "pnl_usd": round(pnl_usd, 2),
                            "pnl_pct": round(pnl_pct, 2),
                            "progress": round(progress, 1),
                            "stop_loss": sl, "tp_min": tp_min, "tp_max": tp_max,
                            "max_loss": round(max_loss, 2),
                            "max_win": round(max_win, 2),
                            "price_history": [{"t": s.get("t", 0), "p": s.get("p", 0), "w": s.get("w", 50)} for s in price_snapshots[-30:]],
                            "strategy_id": p.get("strategy_id", "ensemble"),
                            "bet_mode": p.get("bet_mode", "one_off"),
                            "trade_mode": p.get("trade_mode", "paper"),
                            "est_fee": p.get("est_fee", round(cost_usd * 0.006, 2)),
                            "entry_time": p.get("entry_time", 0),
                            "trailing_stop_pct": trailing_pct,
                            "highest_price_seen": highest_seen,
                            "trailing_sl_price": round(trailing_sl, 2) if trailing_pct > 0 else 0,
                            "strategy_reason": p.get("strategy_reason", ""),
                        })

                # --- Portfolio stats per mode (paper + real) ---
                all_trades = portfolio.get("trade_history", [])
                cash = portfolio.get("cash", 0)
                initial = portfolio.get("initial_balance", 0)

                stats_by_mode = {}
                for mode in ("paper", "real"):
                    m_pos = [p for p in pos_list if p.get("trade_mode", "paper") == mode]
                    m_trades = [t for t in all_trades if t.get("trade_mode", "paper") == mode]
                    m_realized = sum(t.get("pnl_usd", 0) for t in m_trades)
                    m_unrealized = sum(p.get("pnl_usd", 0) for p in m_pos)
                    m_total_pnl = m_realized + m_unrealized
                    m_invested = sum(p.get("cost_usd", 0) for p in m_pos)
                    m_total_trades = len(m_trades)
                    m_wins = sum(1 for t in m_trades if t.get("pnl_usd", 0) > 0)
                    m_win_rate = (m_wins / m_total_trades * 100) if m_total_trades > 0 else 0
                    stats_by_mode[mode] = {
                        "portfolio_value": round(cash + m_invested + m_unrealized, 2),
                        "cash": round(cash, 2),
                        "total_invested": round(m_invested, 2),
                        "total_pnl": round(m_total_pnl, 2),
                        "realized_pnl": round(m_realized, 2),
                        "unrealized_pnl": round(m_unrealized, 2),
                        "total_pnl_pct": round((m_total_pnl / initial * 100) if initial > 0 else 0, 2),
                        "win_rate": round(m_win_rate, 1),
                        "total_trades": m_total_trades,
                        "open_count": len(m_pos),
                    }

                await ws.send_json({
                    "type": "update",
                    "positions": pos_list,
                    "stats": stats_by_mode,
                    "segments": _build_segment_statuses(portfolio, pos_list),
                })

                await asyncio.sleep(2)

            except WebSocketDisconnect:
                break
            except Exception:
                await asyncio.sleep(2)

    except (WebSocketDisconnect, asyncio.TimeoutError, Exception):
        pass
    finally:
        _ws_connections.discard(ws)


@app.get("/api/ticker")
async def api_ticker(request: Request):
    """Live prices for the scrolling ticker bar."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    prices = await _get_cached_prices()
    return prices


@app.get("/api/chart/{symbol}")
async def api_chart(request: Request, symbol: str, timeframe: str = "1h"):
    """OHLCV candles for the price chart."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    # URL uses dash: BTC-USDT -> BTC/USDT
    symbol = symbol.replace("-", "/")
    loop = asyncio.get_event_loop()
    candles = await loop.run_in_executor(None, _fetch_ohlcv_sync, symbol, timeframe)
    return candles


@app.get("/api/strategies")
async def api_strategies(request: Request):
    """Strategy catalog with metadata."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return get_strategy_catalog()


@app.get("/api/signal/{symbol}")
async def api_signal(request: Request, symbol: str, strategy: str = "smart_auto"):
    """AI signal prediction for a symbol using a specific strategy."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    symbol = symbol.replace("-", "/")
    # Cache key includes strategy
    cache_key = f"{symbol}:{strategy}"
    now = time.time()
    cached = _signal_cache.get(cache_key)
    if cached and (now - cached["time"]) < _SIGNAL_CACHE_TTL:
        return cached["data"]
    # Fetch fresh signal with the correct strategy
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, _fetch_signal_sync, symbol, strategy)
    _signal_cache[cache_key] = {"data": data, "time": now}
    return data


# Reasoning cache: {"pos_id": {"data": {...}, "time": float}}
_reasoning_cache: dict = {}
_REASONING_CACHE_TTL = 60  # seconds — heavier analysis, cache longer


def _fetch_reasoning_sync(symbol: str, strategy_id: str, entry_price: float,
                          current_price: float, trailing_pct: float,
                          highest_seen: float) -> dict:
    """Generate detailed AI reasoning for an open position."""
    try:
        config = _SIGNAL_STRATEGY_MAP.get(strategy_id)
        if not config or config["strategy"] is None:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            return {
                "action": "HOLD",
                "summary": "Set & Forget — monitoring SL/TP only, no AI analysis.",
                "details": [],
            }

        ex = _get_public_exchange()
        timeframe = config["timeframe"]

        if timeframe == "multi":
            snapshot = get_multi_tf_snapshot(ex, symbol)
            try:
                ob = get_order_book_pressure(ex, symbol)
                snapshot["buy_pressure"] = ob["buy_pressure"]
                snapshot["spread_pct"] = ob["spread_pct"]
            except Exception:
                snapshot["buy_pressure"] = 0.5
            try:
                snapshot["fear_greed"] = fetch_fear_greed()
            except Exception:
                snapshot["fear_greed"] = 50
            # Enrich with whale + sentiment for AI reasoning
            try:
                from vesper.market_data import enrich_with_intelligence
                enrich_with_intelligence(ex, symbol, snapshot)
            except Exception:
                pass
        else:
            snapshot = get_market_snapshot(ex, symbol, timeframe=timeframe)

        strategy_instance = config["strategy"]()
        result = strategy_instance.analyze(snapshot)

        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        details = []

        # Price action
        details.append(f"Price ${current_price:,.2f} ({pnl_pct:+.2f}% from entry ${entry_price:,.2f})")

        # Trailing stop info
        if trailing_pct > 0 and highest_seen > 0:
            trail_sl = highest_seen * (1 - trailing_pct / 100)
            distance_to_trail = ((current_price - trail_sl) / current_price) * 100
            details.append(f"Trailing SL at ${trail_sl:,.2f} (peak ${highest_seen:,.2f}, {distance_to_trail:.1f}% buffer)")

        # Trend indicators
        rsi = snapshot.get("rsi", 0)
        if rsi > 0:
            rsi_label = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
            details.append(f"RSI {rsi:.0f} ({rsi_label})")

        ema12 = snapshot.get("ema_12", 0)
        ema26 = snapshot.get("ema_26", 0)
        if ema12 and ema26:
            ema_dir = "bullish" if ema12 > ema26 else "bearish"
            details.append(f"EMA 12/26 {ema_dir} (12: ${ema12:,.2f}, 26: ${ema26:,.2f})")

        macd = snapshot.get("macd", 0)
        macd_signal = snapshot.get("macd_signal", 0)
        if macd and macd_signal:
            macd_dir = "bullish" if macd > macd_signal else "bearish"
            details.append(f"MACD {macd_dir} ({macd:.4f} vs signal {macd_signal:.4f})")

        # Multi-TF alignment (for enhanced strategies)
        tf_alignment = snapshot.get("tf_alignment")
        if tf_alignment is not None:
            align_label = "aligned bullish" if tf_alignment == 1.0 else "aligned bearish" if tf_alignment == 0.0 else "mixed signals"
            details.append(f"Multi-TF (1h+4h): {align_label}")

        # Order book
        buy_pressure = snapshot.get("buy_pressure")
        if buy_pressure is not None and buy_pressure != 0.5:
            pressure_label = "buy pressure" if buy_pressure > 0.55 else "sell pressure" if buy_pressure < 0.45 else "balanced"
            details.append(f"Order book: {pressure_label} ({buy_pressure:.0%} buy)")

        # Sentiment
        fear_greed = snapshot.get("fear_greed")
        if fear_greed is not None:
            fg_label = "Extreme Fear" if fear_greed < 25 else "Fear" if fear_greed < 45 else "Extreme Greed" if fear_greed > 75 else "Greed" if fear_greed > 55 else "Neutral"
            details.append(f"Market sentiment: {fg_label} ({fear_greed}/100)")

        # ADX
        adx = snapshot.get("adx", 0)
        if adx > 0:
            strength = "strong trend" if adx > 25 else "weak/no trend"
            details.append(f"ADX {adx:.0f} ({strength})")

        # Whale activity
        whale_score = snapshot.get("whale_score")
        if whale_score is not None and abs(whale_score) > 0.05:
            direction = "bullish (accumulating)" if whale_score > 0 else "bearish (distributing)"
            details.append(f"Whale activity: {direction} ({whale_score:+.2f})")
        whale_details = snapshot.get("whale_details", [])
        for wd in whale_details[:2]:
            details.append(f"  {wd}")

        # Composite sentiment
        sentiment_score = snapshot.get("sentiment_score")
        if sentiment_score is not None and abs(sentiment_score) > 0.05:
            direction = "bullish" if sentiment_score > 0 else "bearish"
            details.append(f"Crowd sentiment: {direction} ({sentiment_score:+.2f})")
        sentiment_details = snapshot.get("sentiment_details", [])
        for sd in sentiment_details[:3]:
            details.append(f"  {sd}")

        # Decision summary
        sig = result.signal.value
        conf = result.confidence
        if sig == "BUY" and pnl_pct >= 0:
            action = "HOLD"
            summary = f"Signal still bullish ({conf:.0%} confidence). Holding position — {result.reason}"
        elif sig == "BUY" and pnl_pct < 0:
            action = "HOLD"
            summary = f"Signal bullish ({conf:.0%}), expecting recovery — {result.reason}"
        elif sig == "SELL":
            action = "CAUTION"
            summary = f"Signal turned bearish ({conf:.0%}). SL/TP will auto-close — {result.reason}"
        else:
            action = "HOLD"
            summary = f"Neutral signal ({conf:.0%}). Waiting for clearer direction — {result.reason}"

        return {
            "action": action,
            "signal": sig,
            "confidence": round(conf, 2),
            "summary": summary,
            "details": details,
            "timeframe": timeframe if timeframe != "multi" else "1h+4h",
        }

    except Exception as e:
        return {
            "action": "HOLD",
            "summary": f"Analysis unavailable — {str(e)[:80]}",
            "details": [],
        }


@app.get("/api/reasoning")
async def api_reasoning(request: Request, pid: str = ""):
    """Live AI reasoning for why a position is being held or should close."""
    position_id = pid
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    # Load portfolio to find the position
    portfolio = _load_portfolio(user.id)
    pos = portfolio.get("positions", {}).get(position_id)
    if not pos:
        return JSONResponse({"error": "Position not found"}, status_code=404)

    # Check cache
    now = time.time()
    cached = _reasoning_cache.get(position_id)
    if cached and (now - cached["time"]) < _REASONING_CACHE_TTL:
        return cached["data"]

    # Get current price
    prices = await _get_cached_prices()
    price_map = {p["symbol"]: p["price"] for p in prices}
    current_price = price_map.get(pos["symbol"], pos["entry_price"])

    # Run analysis
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(
        None, _fetch_reasoning_sync,
        pos["symbol"],
        pos.get("strategy_id", "smart_auto"),
        pos["entry_price"],
        current_price,
        pos.get("trailing_stop_pct", 0),
        pos.get("highest_price_seen", 0),
    )
    _reasoning_cache[position_id] = {"data": data, "time": now}
    return data


@app.get("/api/portfolio-stats")
async def api_portfolio_stats(request: Request, mode: str = ""):
    """Live portfolio stats: value, P&L (realized + unrealized), win rate."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    portfolio = _load_portfolio(user.id)
    cash = portfolio.get("cash", user.paper_balance)
    initial = portfolio.get("initial_balance", user.paper_balance)
    all_positions = portfolio.get("positions", {})
    all_trades = portfolio.get("trade_history", [])

    # Filter by mode (from query param, or fallback to user's server-side setting)
    if mode in ("paper", "real"):
        current_mode = mode
    else:
        mode_match = {"live": "real", "paper": "paper"}
        current_mode = mode_match.get(user.trading_mode, "paper")
    positions = {
        pid: p for pid, p in all_positions.items()
        if p.get("trade_mode", "paper") == current_mode
    }
    trades = [
        t for t in all_trades
        if t.get("trade_mode", "paper") == current_mode
    ]

    # Realized P&L from closed trades
    realized_pnl = sum(t.get("pnl_usd", 0) for t in trades)
    wins = sum(1 for t in trades if t.get("pnl_usd", 0) > 0)
    total_trades = len(trades)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    # Unrealized P&L from open positions
    unrealized_pnl = 0
    if positions:
        pos_symbols = {p.get("symbol", "") for p in positions.values() if not p.get("symbol", "").startswith("PRED:")}
        price_map = await _get_price_map_for_positions(pos_symbols, user)
        for p in positions.values():
            entry = p.get("entry_price", 0)
            sym = p.get("symbol", "")
            if sym.startswith("PRED:"):
                current = p.get("current_probability", entry)
            else:
                current = price_map.get(sym, entry)
            amount = p.get("amount", 0)
            side = p.get("side", "buy")
            if side == "buy":
                unrealized_pnl += (current - entry) * amount
            else:
                unrealized_pnl += (entry - current) * amount

    total_pnl = realized_pnl + unrealized_pnl
    portfolio_value = cash + sum(
        p.get("cost_usd", 0) for p in positions.values()
    ) + unrealized_pnl

    # P&L history for chart: each closed trade as a cumulative data point
    pnl_history = []
    running_pnl = 0
    for t in trades:
        running_pnl += t.get("pnl_usd", 0)
        pnl_history.append({
            "time": t.get("exit_time", 0),
            "pnl": round(running_pnl, 2),
            "trade_pnl": round(t.get("pnl_usd", 0), 2),
            "symbol": t.get("symbol", ""),
        })

    # Total invested = sum of all open position costs
    total_invested = sum(p.get("cost_usd", 0) for p in positions.values())

    return {
        "portfolio_value": round(portfolio_value, 2),
        "cash": round(cash, 2),
        "initial_balance": round(initial, 2),
        "total_invested": round(total_invested, 2),
        "total_pnl": round(total_pnl, 2),
        "realized_pnl": round(realized_pnl, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "total_pnl_pct": round((total_pnl / initial * 100) if initial > 0 else 0, 2),
        "win_rate": round(win_rate, 1),
        "total_trades": total_trades,
        "open_count": len(positions),
        "pnl_history": pnl_history,
    }


@app.get("/api/positions")
async def api_positions(request: Request):
    """Open positions with live unrealized P&L."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    portfolio = _load_portfolio(user.id)
    positions = portfolio.get("positions", {})

    if not positions:
        return []

    # Build price map that covers ALL position symbols (not just TICKER_SYMBOLS)
    pos_symbols = {p.get("symbol", "") for p in positions.values() if not p.get("symbol", "").startswith("PRED:")}
    price_map = await _get_price_map_for_positions(pos_symbols, user)

    result = []
    for pid, p in positions.items():
        sym = p.get("symbol", "")
        entry = p.get("entry_price", 0)
        # For prediction positions (PRED:*), use stored probability data for P&L
        is_prediction = sym.startswith("PRED:")
        if is_prediction:
            # Entry price = probability at entry (e.g. 0.35 means 35%)
            # Current value = latest probability or stored estimate
            stored_prob = p.get("current_probability", entry)
            current = stored_prob
        else:
            current = price_map.get(sym, entry)
        amount = p.get("amount", 0)
        side = p.get("side", "buy")

        if side == "buy":
            pnl_usd = (current - entry) * amount
            pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
        else:
            pnl_usd = (entry - current) * amount
            pnl_pct = ((entry - current) / entry * 100) if entry > 0 else 0

        limits = p.get("limits", {})
        sl = limits.get("stop_loss_price", 0)
        tp_max = limits.get("take_profit_max_price", 0)
        # Progress: 0% at stop-loss, 100% at take-profit
        price_range = tp_max - sl if tp_max > sl else 1
        progress = max(0, min(100, ((current - sl) / price_range) * 100))

        cost_usd = p.get("cost_usd", 0)
        tp_min = limits.get("take_profit_min_price", 0)
        # Max loss/win in dollar terms
        if side == "buy":
            max_loss_usd = (entry - sl) * amount if sl > 0 else cost_usd
            max_win_usd = (tp_max - entry) * amount if tp_max > 0 else 0
        else:
            max_loss_usd = (sl - entry) * amount if sl > 0 else cost_usd
            max_win_usd = (entry - tp_max) * amount if tp_max > 0 else 0
        max_loss_usd = abs(max_loss_usd)
        max_win_usd = abs(max_win_usd)

        # Win probability: how far is price from SL vs TP
        # 0% at SL, 100% at TP max, interpolated linearly
        trailing_sl = 0
        trailing_pct = p.get("trailing_stop_pct", 0)
        highest_seen = p.get("highest_price_seen", 0)
        if trailing_pct > 0 and highest_seen > 0:
            trailing_sl = highest_seen * (1 - trailing_pct / 100)

        effective_sl = max(sl, trailing_sl) if trailing_sl > 0 else sl
        if side == "buy":
            range_total = tp_max - effective_sl if tp_max > effective_sl else 1
            win_prob = max(0, min(100, ((current - effective_sl) / range_total) * 100))
        else:
            range_total = effective_sl - tp_max if effective_sl > tp_max else 1
            win_prob = max(0, min(100, ((effective_sl - current) / range_total) * 100))

        # Price history snapshots for mini sparkline
        price_snapshots = p.get("price_history", [])
        # Store new snapshot (kept lean — max 60 points)
        now_ts = time.time()
        if len(price_snapshots) == 0 or (now_ts - (price_snapshots[-1].get("t", 0))) > 30:
            price_snapshots.append({"t": now_ts, "p": current, "w": round(win_prob, 1)})
            if len(price_snapshots) > 60:
                price_snapshots = price_snapshots[-60:]
            p["price_history"] = price_snapshots
            _save_portfolio(user.id, portfolio)

        result.append({
            "id": pid,
            "symbol": sym,
            "name": sym.split("/")[0] if sym else "",
            "side": side,
            "entry_price": entry,
            "current_price": current,
            "cost_usd": cost_usd,
            "amount": amount,
            "pnl_usd": round(pnl_usd, 2),
            "pnl_pct": round(pnl_pct, 2),
            "progress": round(progress, 1),
            "stop_loss": sl,
            "tp_min": tp_min,
            "tp_max": tp_max,
            "max_loss": round(max_loss_usd, 2),
            "max_win": round(max_win_usd, 2),
            "win_probability": round(win_prob, 1),
            "price_history": [{"t": s.get("t", 0), "p": s.get("p", 0), "w": s.get("w", 50)} for s in price_snapshots[-30:]],
            "strategy_id": p.get("strategy_id", "ensemble"),
            "bet_mode": p.get("bet_mode", "one_off"),
            "trade_mode": p.get("trade_mode", "paper"),
            "est_fee": p.get("est_fee", round(cost_usd * 0.006, 2)),
            "entry_time": p.get("entry_time", 0),
            "trailing_stop_pct": trailing_pct,
            "highest_price_seen": highest_seen,
            "trailing_sl_price": round(trailing_sl, 2) if trailing_pct > 0 else 0,
            "strategy_reason": p.get("strategy_reason", ""),
        })
    return result


@app.post("/api/open-trade")
async def api_open_trade(request: Request):
    """Open a new manual trade from the dashboard."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    body = await request.json()
    symbol = body.get("symbol", "")
    strategy_id = body.get("strategy_id", "set_and_forget")
    amount_usd = float(body.get("amount_usd", 0))
    stop_loss_pct = float(body.get("stop_loss_pct", 2.0))
    tp_min_pct = float(body.get("tp_min_pct", 1.5))
    tp_max_pct = float(body.get("tp_max_pct", 5.0))
    trailing_stop_pct = float(body.get("trailing_stop_pct", 0))
    bet_mode = body.get("bet_mode", "one_off")
    trade_mode = body.get("trade_mode", "paper")

    if not symbol or amount_usd <= 0:
        return JSONResponse({"error": "Invalid symbol or amount"}, status_code=400)

    # Validate real mode requires API keys
    if trade_mode == "real" and not user.coinbase_api_key:
        return JSONResponse({"error": "Add Coinbase API keys in Settings to place real trades"}, status_code=400)

    # Load portfolio
    portfolio_path = os.path.join(DATA_DIR, f"portfolio_{user.id}.json")
    if os.path.exists(portfolio_path):
        with open(portfolio_path) as f:
            pdata = json.load(f)
    else:
        pdata = {"cash": user.paper_balance, "initial_balance": user.paper_balance,
                 "positions": {}, "trade_history": []}

    cash = pdata.get("cash", 0)
    if amount_usd > cash:
        return JSONResponse({"error": f"Insufficient funds. Available: ${cash:.2f}"}, status_code=400)

    # Get current price
    prices = await _get_cached_prices()
    price_map = {p["symbol"]: p["price"] for p in prices}
    current_price = price_map.get(symbol, 0)
    if current_price <= 0:
        return JSONResponse({"error": "Could not fetch price for " + symbol}, status_code=400)

    # Calculate position
    asset_amount = amount_usd / current_price
    # Coinbase Advanced taker fee ~0.60%
    fee_rate = 0.006
    est_fee = round(amount_usd * fee_rate, 2)

    # Place real order if trade_mode is real
    if trade_mode == "real":
        try:
            from config.settings import ExchangeConfig, is_stock_symbol
            if is_stock_symbol(symbol) and user.has_alpaca:
                from vesper.exchange import AlpacaExchange, alpaca_market_buy
                alpaca = AlpacaExchange(
                    api_key=user.get_alpaca_key(),
                    api_secret=user.get_alpaca_secret(),
                    paper=False,
                )
                order = alpaca_market_buy(alpaca, symbol, amount_usd)
                current_price = float(order.get("average", current_price))
                asset_amount = float(order.get("filled", asset_amount))
                amount_usd = current_price * asset_amount
            else:
                from vesper.exchange import create_exchange, place_market_buy
                cfg = ExchangeConfig(
                    api_key=user.get_api_key(),
                    api_secret=user.get_api_secret(),
                    sandbox=False,
                )
                exchange = create_exchange(cfg)
                order = place_market_buy(exchange, symbol, asset_amount)
                current_price = float(order.get("average", current_price))
                asset_amount = float(order.get("filled", asset_amount))
                amount_usd = current_price * asset_amount
                order_fee = order.get("fee", {})
                if order_fee and order_fee.get("cost"):
                    est_fee = round(float(order_fee["cost"]), 2)
        except Exception as e:
            return JSONResponse({"error": f"Exchange error: {str(e)}"}, status_code=400)

    stop_loss_price = current_price * (1 - stop_loss_pct / 100)
    tp_min_price = current_price * (1 + tp_min_pct / 100)
    # When trailing is active, disable fixed TP max (let it run)
    tp_max_price = current_price * 100 if trailing_stop_pct > 0 else current_price * (1 + tp_max_pct / 100)

    pos_id = f"{symbol}-{int(time.time())}"
    position = {
        "symbol": symbol,
        "side": "buy",
        "entry_price": current_price,
        "amount": asset_amount,
        "cost_usd": amount_usd,
        "entry_time": time.time(),
        "strategy_reason": f"Manual: {strategy_id}",
        "id": pos_id,
        "strategy_id": strategy_id,
        "bet_mode": bet_mode,
        "trade_mode": trade_mode,
        "stop_loss_pct": stop_loss_pct,
        "tp_min_pct": tp_min_pct,
        "tp_max_pct": tp_max_pct,
        "trailing_stop_pct": trailing_stop_pct,
        "highest_price_seen": current_price,
        "est_fee": est_fee,
        "limits": {
            "stop_loss_price": stop_loss_price,
            "take_profit_min_price": tp_min_price,
            "take_profit_max_price": tp_max_price,
            "position_size_usd": amount_usd,
            "position_size_asset": asset_amount,
            "trailing_stop_pct": trailing_stop_pct,
            "highest_price_seen": current_price,
        },
    }

    # Save
    pdata["cash"] = cash - amount_usd
    pdata["positions"][pos_id] = position
    with open(portfolio_path, "w") as f:
        json.dump(pdata, f, indent=2)

    return {"ok": True, "position_id": pos_id, "entry_price": current_price, "fee": est_fee}


@app.post("/api/close-trade")
async def api_close_trade(request: Request):
    """Manually close an open position."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    body = await request.json()
    position_id = body.get("position_id", "")

    portfolio_path = os.path.join(DATA_DIR, f"portfolio_{user.id}.json")
    if not os.path.exists(portfolio_path):
        return JSONResponse({"error": "No portfolio found"}, status_code=404)

    with open(portfolio_path) as f:
        pdata = json.load(f)

    pos = pdata.get("positions", {}).get(position_id)
    if not pos:
        return JSONResponse({"error": "Position not found"}, status_code=404)

    # Get current price
    sym = pos["symbol"]
    price_map = await _get_price_map_for_positions({sym}, user)
    current_price = price_map.get(sym, pos["entry_price"])

    # Calculate P&L
    entry = pos["entry_price"]
    amount = pos["amount"]
    if pos.get("side", "buy") == "buy":
        pnl_usd = (current_price - entry) * amount
        pnl_pct = ((current_price - entry) / entry * 100) if entry > 0 else 0
    else:
        pnl_usd = (entry - current_price) * amount
        pnl_pct = ((entry - current_price) / entry * 100) if entry > 0 else 0

    # Record trade (preserve trade_mode for siloing)
    trade = {
        "symbol": pos["symbol"],
        "side": pos.get("side", "buy"),
        "entry_price": entry,
        "exit_price": current_price,
        "amount": amount,
        "pnl_usd": round(pnl_usd, 2),
        "pnl_pct": round(pnl_pct, 2),
        "entry_time": pos.get("entry_time", 0),
        "exit_time": time.time(),
        "reason": "Manual close",
        "strategy_reason": pos.get("strategy_reason", ""),
        "trade_mode": pos.get("trade_mode", "paper"),
    }

    pdata["cash"] = pdata.get("cash", 0) + pos.get("cost_usd", 0) + pnl_usd
    pdata.setdefault("trade_history", []).append(trade)
    del pdata["positions"][position_id]

    with open(portfolio_path, "w") as f:
        json.dump(pdata, f, indent=2)

    return {
        "ok": True,
        "symbol": pos["symbol"],
        "entry_price": entry,
        "exit_price": current_price,
        "pnl_usd": round(pnl_usd, 2),
        "pnl_pct": round(pnl_pct, 2),
        "cost_usd": pos.get("cost_usd", 0),
    }


@app.get("/api/decisions")
async def api_decisions(request: Request, limit: int = 50):
    """Return the bot's structured decision log — why it entered, exited, or skipped trades."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    portfolio = _load_portfolio(user.id)
    decisions = portfolio.get("decision_log", [])
    # Return most recent first
    return {"decisions": list(reversed(decisions[-limit:]))}


# ═══════════════════════════════════════
# Autopilot API
# ═══════════════════════════════════════

@app.get("/api/autopilot")
async def api_autopilot_status(request: Request):
    """Get autopilot status and positions."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    portfolio = _load_portfolio(user.id)
    autopilot = portfolio.get("autopilot", {})
    positions = portfolio.get("positions", {})

    # Autopilot positions
    ap_positions = []
    deployed = 0.0
    pos_symbols = set()
    for pid, p in positions.items():
        if p.get("strategy_id") == "autopilot":
            deployed += p.get("cost_usd", 0)
            sym = p.get("symbol", "")
            pos_symbols.add(sym)
            ap_positions.append({
                "id": pid,
                "symbol": sym,
                "entry_price": p.get("entry_price", 0),
                "cost_usd": p.get("cost_usd", 0),
                "entry_time": p.get("entry_time", 0),
            })

    fund_total = autopilot.get("fund_usd", 0)

    # Compute unrealized P&L
    unrealized_pnl = 0.0
    if ap_positions:
        price_map = await _get_price_map_for_positions(pos_symbols, user)
        for ap in ap_positions:
            current = price_map.get(ap["symbol"], ap["entry_price"])
            if ap["entry_price"] > 0 and ap["cost_usd"] > 0:
                amount = ap["cost_usd"] / ap["entry_price"]
                unrealized_pnl += (current - ap["entry_price"]) * amount

    # Compute realized P&L
    all_trades = portfolio.get("trade_history", [])
    realized_pnl = sum(
        t.get("pnl_usd", 0) for t in all_trades
        if t.get("strategy_id") == "autopilot"
    )

    return {
        "enabled": autopilot.get("enabled", False),
        "fund_usd": fund_total,
        "deployed_usd": round(deployed, 2),
        "available_usd": round(fund_total - deployed, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "realized_pnl": round(realized_pnl, 2),
        "max_positions": autopilot.get("max_positions", 20),
        "max_bet_usd": autopilot.get("max_bet_usd", 0),
        "risk_level": autopilot.get("risk_level", "aggressive"),
        "reinvest_pct": autopilot.get("reinvest_pct", 100),
        "positions": ap_positions,
        "log": portfolio.get("autopilot_log", [])[-20:],
    }


@app.post("/api/autopilot")
async def api_autopilot_toggle(request: Request):
    """Enable/disable autopilot or update fund amount."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    body = await request.json()
    action = body.get("action", "")
    amount = float(body.get("amount_usd", 0))

    portfolio_path = os.path.join(
        os.environ.get("VESPER_DATA_DIR", "data"),
        f"portfolio_{user.id}.json",
    )
    pdata = {}
    if os.path.exists(portfolio_path):
        with open(portfolio_path) as f:
            pdata = json.load(f)

    if action == "start":
        if amount < 10:
            return JSONResponse({"error": "Minimum $10"}, status_code=400)
        pdata["autopilot"] = {
            "enabled": True,
            "fund_usd": amount,
            "max_positions": int(body.get("max_positions", 20)),
            "risk_level": body.get("risk_level", "aggressive"),
            "reinvest_pct": float(body.get("reinvest_pct", 100)),
            "trade_mode": body.get("trade_mode", "paper"),
            "started_at": time.time(),
        }
    elif action == "stop":
        if "autopilot" in pdata:
            pdata["autopilot"]["enabled"] = False
    elif action == "update":
        if "autopilot" not in pdata:
            pdata["autopilot"] = {}
        if amount > 0:
            pdata["autopilot"]["fund_usd"] = amount
        mp = body.get("max_positions")
        if mp:
            pdata["autopilot"]["max_positions"] = int(mp)
        risk = body.get("risk_level")
        if risk in ("conservative", "moderate", "aggressive"):
            pdata["autopilot"]["risk_level"] = risk
        rp = body.get("reinvest_pct")
        if rp is not None:
            pdata["autopilot"]["reinvest_pct"] = float(rp)
        mb = body.get("max_bet_usd")
        if mb is not None:
            pdata["autopilot"]["max_bet_usd"] = float(mb)

    with open(portfolio_path, "w") as f:
        json.dump(pdata, f, indent=2)

    return {"ok": True, "autopilot": pdata.get("autopilot", {})}


# ═══════════════════════════════════════
# Altcoin Hunter API
# ═══════════════════════════════════════

@app.get("/api/altcoin-hunter")
async def api_altcoin_hunter_status(request: Request):
    """Get altcoin hunter status and positions."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    portfolio = _load_portfolio(user.id)
    hunter = portfolio.get("altcoin_hunter", {})
    positions = portfolio.get("positions", {})

    hunter_positions = []
    deployed = 0.0
    pos_symbols = set()
    for pid, p in positions.items():
        if p.get("strategy_id") == "altcoin_hunter":
            deployed += p.get("cost_usd", 0)
            sym = p.get("symbol", "")
            pos_symbols.add(sym)
            hunter_positions.append({
                "id": pid,
                "symbol": sym,
                "entry_price": p.get("entry_price", 0),
                "cost_usd": p.get("cost_usd", 0),
                "entry_time": p.get("entry_time", 0),
                "strategy_reason": p.get("strategy_reason", ""),
            })

    fund_total = hunter.get("fund_usd", 0)

    # Compute unrealized P&L from open positions
    unrealized_pnl = 0.0
    if hunter_positions:
        price_map = await _get_price_map_for_positions(pos_symbols, user)
        for hp in hunter_positions:
            current = price_map.get(hp["symbol"], hp["entry_price"])
            if hp["entry_price"] > 0 and hp["cost_usd"] > 0:
                amount = hp["cost_usd"] / hp["entry_price"]
                unrealized_pnl += (current - hp["entry_price"]) * amount

    # Compute realized P&L from closed trades
    all_trades = portfolio.get("trade_history", [])
    realized_pnl = sum(
        t.get("pnl_usd", 0) for t in all_trades
        if t.get("strategy_id") == "altcoin_hunter" or "Altcoin Hunter" in t.get("reason", "")
    )

    # Get recent altcoin_hunter scan logs
    all_logs = portfolio.get("autopilot_log", [])
    hunter_logs = [l for l in all_logs if l.get("type") == "altcoin_hunter_scan"][-10:]

    return {
        "enabled": hunter.get("enabled", False),
        "fund_usd": fund_total,
        "deployed_usd": round(deployed, 2),
        "available_usd": round(fund_total - deployed, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "realized_pnl": round(realized_pnl, 2),
        "max_positions": hunter.get("max_positions", 20),
        "max_bet_usd": hunter.get("max_bet_usd", 0),
        "trailing_stop_pct": hunter.get("trailing_stop_pct", 2.0),
        "risk_level": hunter.get("risk_level", "aggressive"),
        "reinvest_pct": hunter.get("reinvest_pct", 100),
        "positions": hunter_positions,
        "log": hunter_logs,
    }


@app.post("/api/altcoin-hunter")
async def api_altcoin_hunter_toggle(request: Request):
    """Enable/disable altcoin hunter or update config."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    body = await request.json()
    action = body.get("action", "")
    amount = float(body.get("amount_usd", 0))

    portfolio_path = os.path.join(
        os.environ.get("VESPER_DATA_DIR", "data"),
        f"portfolio_{user.id}.json",
    )
    pdata = {}
    if os.path.exists(portfolio_path):
        with open(portfolio_path) as f:
            pdata = json.load(f)

    if action == "start":
        if amount < 10:
            return JSONResponse({"error": "Minimum $10"}, status_code=400)
        pdata["altcoin_hunter"] = {
            "enabled": True,
            "fund_usd": amount,
            "max_positions": int(body.get("max_positions", 5)),
            "trailing_stop_pct": float(body.get("trailing_stop_pct", 2.0)),
            "risk_level": body.get("risk_level", "aggressive"),
            "reinvest_pct": float(body.get("reinvest_pct", 100)),
            "trade_mode": body.get("trade_mode", "paper"),
            "started_at": time.time(),
        }
    elif action == "stop":
        if "altcoin_hunter" in pdata:
            pdata["altcoin_hunter"]["enabled"] = False
    elif action == "update":
        if "altcoin_hunter" not in pdata:
            pdata["altcoin_hunter"] = {}
        if amount > 0:
            pdata["altcoin_hunter"]["fund_usd"] = amount
        risk = body.get("risk_level")
        if risk in ("conservative", "moderate", "aggressive"):
            pdata["altcoin_hunter"]["risk_level"] = risk
        rp = body.get("reinvest_pct")
        if rp is not None:
            pdata["altcoin_hunter"]["reinvest_pct"] = float(rp)
        mb = body.get("max_bet_usd")
        if mb is not None:
            pdata["altcoin_hunter"]["max_bet_usd"] = float(mb)

    with open(portfolio_path, "w") as f:
        json.dump(pdata, f, indent=2)

    return {"ok": True, "altcoin_hunter": pdata.get("altcoin_hunter", {})}


# ═══════════════════════════════════════
# Predictions Autopilot API
# ═══════════════════════════════════════

@app.get("/api/predictions-autopilot")
async def api_predictions_autopilot_status(request: Request):
    """Get predictions autopilot status and positions."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    portfolio = _load_portfolio(user.id)
    pred_cfg = portfolio.get("predictions_autopilot", {})
    positions = portfolio.get("positions", {})

    pred_positions = []
    deployed = 0.0
    for pid, p in positions.items():
        if p.get("strategy_id") == "predictions":
            deployed += p.get("cost_usd", 0)
            pred_positions.append({
                "id": pid,
                "symbol": p.get("symbol", ""),
                "entry_price": p.get("entry_price", 0),
                "cost_usd": p.get("cost_usd", 0),
                "entry_time": p.get("entry_time", 0),
                "strategy_reason": p.get("strategy_reason", ""),
                "prediction_side": p.get("prediction_side", ""),
                "prediction_question": p.get("prediction_question", ""),
            })

    fund_total = pred_cfg.get("fund_usd", 0)

    # Compute realized P&L from closed predictions
    all_trades = portfolio.get("trade_history", [])
    realized_pnl = sum(
        t.get("pnl_usd", 0) for t in all_trades
        if t.get("strategy_id") == "predictions"
    )

    # Get recent prediction scan logs
    all_logs = portfolio.get("autopilot_log", [])
    pred_logs = [l for l in all_logs if l.get("type") == "predictions_scan"][-10:]

    return {
        "enabled": pred_cfg.get("enabled", False),
        "fund_usd": fund_total,
        "deployed_usd": round(deployed, 2),
        "available_usd": round(fund_total - deployed, 2),
        "unrealized_pnl": 0,
        "realized_pnl": round(realized_pnl, 2),
        "max_positions": pred_cfg.get("max_positions", 20),
        "max_bet_usd": pred_cfg.get("max_bet_usd", 0),
        "risk_level": pred_cfg.get("risk_level", "aggressive"),
        "reinvest_pct": pred_cfg.get("reinvest_pct", 100),
        "positions": pred_positions,
        "log": pred_logs,
        "last_scan_time": pred_cfg.get("last_scan_time", 0),
    }


@app.post("/api/predictions-autopilot")
async def api_predictions_autopilot_toggle(request: Request):
    """Enable/disable predictions autopilot."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    body = await request.json()
    action = body.get("action", "")
    amount = float(body.get("amount_usd", 0))

    portfolio_path = os.path.join(
        os.environ.get("VESPER_DATA_DIR", "data"),
        f"portfolio_{user.id}.json",
    )
    pdata = {}
    if os.path.exists(portfolio_path):
        with open(portfolio_path) as f:
            pdata = json.load(f)

    if action == "start":
        if amount < 10:
            return JSONResponse({"error": "Minimum $10"}, status_code=400)
        pdata["predictions_autopilot"] = {
            "enabled": True,
            "fund_usd": amount,
            "max_positions": int(body.get("max_positions", 20)),
            "min_edge_pct": float(body.get("min_edge_pct", 5.0)),
            "risk_level": body.get("risk_level", "aggressive"),
            "reinvest_pct": float(body.get("reinvest_pct", 100)),
            "trade_mode": body.get("trade_mode", "paper"),
            "started_at": time.time(),
            "last_scan_time": 0,
        }
    elif action == "stop":
        if "predictions_autopilot" in pdata:
            pdata["predictions_autopilot"]["enabled"] = False
    elif action == "update":
        if "predictions_autopilot" not in pdata:
            pdata["predictions_autopilot"] = {}
        if amount > 0:
            pdata["predictions_autopilot"]["fund_usd"] = amount
        risk = body.get("risk_level")
        if risk in ("conservative", "moderate", "aggressive"):
            pdata["predictions_autopilot"]["risk_level"] = risk
        rp = body.get("reinvest_pct")
        if rp is not None:
            pdata["predictions_autopilot"]["reinvest_pct"] = float(rp)
        mb = body.get("max_bet_usd")
        if mb is not None:
            pdata["predictions_autopilot"]["max_bet_usd"] = float(mb)

    with open(portfolio_path, "w") as f:
        json.dump(pdata, f, indent=2)

    return {"ok": True, "predictions_autopilot": pdata.get("predictions_autopilot", {})}


# ═══════════════════════════════════════
# Position Analysis API (Learn More)
# ═══════════════════════════════════════

@app.get("/api/position-analysis/{pid}")
async def api_position_analysis(pid: str, request: Request):
    """Get full AI analysis data for a position (indicators, deep search, reasoning)."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    portfolio = _load_portfolio(user.id)
    analyses = portfolio.get("position_analyses", {})
    analysis = analyses.get(pid, {})
    if not analysis:
        # Build analysis from position metadata as fallback
        pos = portfolio.get("positions", {}).get(pid, {})
        if not pos:
            return {"found": False, "strategy_reason": "Position not found."}
        limits = pos.get("limits", {})
        fallback = {
            "found": True,
            "signal": "BUY" if pos.get("side", "buy") == "buy" else "SELL",
            "confidence": pos.get("confidence", 0),
            "reasoning": pos.get("strategy_reason", ""),
            "indicators": pos.get("indicators", {}),
        }
        # Include search summary if available
        if pos.get("search_summary"):
            fallback["search_summary"] = pos["search_summary"]
        # Build factors from strategy reason text
        reason = pos.get("strategy_reason", "")
        if reason:
            bullish = []
            bearish = []
            for line in reason.replace(";", ".").split("."):
                line = line.strip()
                if not line:
                    continue
                lower = line.lower()
                if any(w in lower for w in ["bullish", "uptrend", "above", "support", "momentum", "surge", "accumulation", "buy", "positive"]):
                    bullish.append(line)
                elif any(w in lower for w in ["bearish", "downtrend", "below", "resistance", "overbought", "sell", "negative", "weak"]):
                    bearish.append(line)
            if bullish:
                fallback["bullish_factors"] = bullish[:5]
            if bearish:
                fallback["bearish_factors"] = bearish[:5]
        return fallback
    return {"found": True, **analysis}


# ═══════════════════════════════════════
# Polymarket API
# ═══════════════════════════════════════

_polymarket_cache: dict = {}  # keyed by max_days


@app.get("/api/polymarket/markets")
async def api_polymarket_markets(request: Request, limit: int = 20, max_days: int = 14):
    """Fetch trending Polymarket prediction markets with AI edge estimation.

    Args:
        limit: Max markets to return (default 20).
        max_days: Only markets expiring within N days (default 14, 0=all).
    """
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    # Cache key includes max_days since different filters = different results
    cache_key = f"{max_days}"
    now = time.time()
    cached = _polymarket_cache.get(cache_key)
    if cached and now - cached["ts"] < 120:
        markets = cached["data"]
    else:
        try:
            from vesper.polymarket import get_trending_markets
            markets = get_trending_markets(limit=50, max_days=max_days)
            _polymarket_cache[cache_key] = {"data": markets, "ts": now}
        except Exception as e:
            return JSONResponse({"error": f"Polymarket API error: {str(e)}"}, status_code=502)

    return {"markets": markets[:limit]}


# ═══════════════════════════════════════
# Kalshi API (Coinbase Predictions source)
# ═══════════════════════════════════════

_kalshi_cache: dict = {}  # keyed by max_days


@app.get("/api/kalshi/markets")
async def api_kalshi_markets(request: Request, limit: int = 20, max_days: int = 0):
    """Fetch Kalshi prediction markets (powers Coinbase predictions).

    Args:
        limit: Max markets to return (default 20).
        max_days: Only markets expiring within N days (default 0=all).
    """
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    cache_key = f"{max_days}"
    now = time.time()
    cached = _kalshi_cache.get(cache_key)
    if cached and now - cached["ts"] < 120:
        markets = cached["data"]
    else:
        try:
            from vesper.kalshi import get_kalshi_markets
            markets = get_kalshi_markets(limit=50, max_days=max_days)
            _kalshi_cache[cache_key] = {"data": markets, "ts": now}
        except Exception as e:
            return JSONResponse({"error": f"Kalshi API error: {str(e)}"}, status_code=502)

    return {"markets": markets[:limit]}


@app.get("/api/research/market")
async def api_research_market(request: Request, question: str = "", market_prob: float = 50, category: str = ""):
    """Deep AI research for a specific prediction market question.

    Uses platform-provided Perplexity + Claude pipeline:
    1. Perplexity searches the web for current information
    2. Claude analyzes evidence and produces a calibrated probability
    """
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    if not question:
        return JSONResponse({"error": "question parameter required"}, status_code=400)

    # Platform keys are used automatically (env vars)
    # No per-user key needed
    pplx_key = os.environ.get("PERPLEXITY_API_KEY", "")
    if not pplx_key:
        return JSONResponse({
            "researched": False,
            "reason": "no_api_key",
            "message": "Research API not configured on this platform.",
        })

    try:
        from vesper.ai_research import research_market
        result = research_market(question, market_prob, category=category)
        return result
    except Exception as e:
        return JSONResponse({"error": f"Research failed: {str(e)}"}, status_code=500)
