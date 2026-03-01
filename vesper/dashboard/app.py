"""Vesper Dashboard — SaaS with OAuth, 2FA, HTTPS."""

import asyncio
import base64
import hashlib
import hmac
import io
import logging
import os
import json
import secrets
import time
import traceback
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
_log = logging.getLogger("vesper.dashboard")

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
    verify_password, update_password, update_api_keys, update_alpaca_keys, update_kalshi_keys,
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
    _log.info("Dashboard starting up — initialising database")
    init_db()
    # Preload CoinGecko price data in background (non-blocking)
    import threading
    threading.Thread(target=_fetch_coingecko_market_caps, daemon=True).start()
    _log.info("Dashboard ready")


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

def _clean_nones(obj):
    """Recursively replace None/null values with sensible defaults.

    JSON stores null which Python reads as None.  When code later does
    ``val = d.get("price", 0)`` the default 0 is *ignored* because the
    key exists (with value None).  Arithmetic on None then crashes with
    TypeError.  Cleaning at load-time prevents this class of bug globally.
    """
    if isinstance(obj, dict):
        return {k: _clean_nones(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nones(v) for v in obj]
    if obj is None:
        return 0
    return obj


def _s(val) -> str:
    """Safe string coercion — portfolio values may be 0 (int) when JSON had null.
    Always returns a string so .startswith(), ``in``, .split() etc. never crash."""
    if isinstance(val, str):
        return val
    if val is None or val == 0:
        return ""
    return str(val)


def _load_portfolio(uid: int) -> dict:
    p = os.path.join(DATA_DIR, f"portfolio_{uid}.json")
    if not os.path.exists(p):
        return {"cash": 0, "initial_balance": 0, "positions": {}, "trade_history": []}
    with open(p) as f:
        return _clean_nones(json.load(f))


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
async def login_page(request: Request, error: str = "", need_2fa: str = ""):
    # If a trust cookie exists, the device is likely remembered — hide 2FA field
    has_trust = bool(request.cookies.get("vesper_trust", ""))
    # But if we redirected back with need_2fa=1, show it anyway (cookie was expired)
    show_2fa = not has_trust or need_2fa == "1"
    return templates.TemplateResponse("login.html", {
        "request": request, "error": error,
        "locked": _is_locked(_get_ip(request)),
        "show_2fa": show_2fa,
        **_oauth_context(),
    })

@app.post("/login")
async def login(request: Request, email: str = Form(...),
                password: str = Form(...), totp_code: str = Form(""),
                remember_me: str = Form("")):
    ip = _get_ip(request)
    _log.info(f"[login] attempt email={email} ip={ip}")
    if _is_locked(ip):
        _log.warning(f"[login] locked ip={ip}")
        return RedirectResponse(url="/login?error=locked", status_code=303)
    user = get_user_by_email(email)
    if not user or not verify_password(user, password):
        _log.warning(f"[login] invalid credentials email={email} ip={ip}")
        _record_fail(ip)
        return RedirectResponse(url="/login?error=invalid", status_code=303)

    # Check if this device is trusted (skip 2FA)
    trust_cookie = request.cookies.get("vesper_trust", "")
    device_trusted = False
    if trust_cookie:
        token_hash = hashlib.sha256(trust_cookie.encode()).hexdigest()
        device_trusted = is_device_trusted(user.id, token_hash)

    if not device_trusted:
        if not totp_code:
            # Device was expected to be trusted but cookie is invalid/expired
            # Redirect back to login with 2FA field visible
            _log.info(f"[login] trust cookie expired, requesting 2FA email={email}")
            return RedirectResponse(url="/login?need_2fa=1", status_code=303)
        if not pyotp.TOTP(user.totp_secret).verify(totp_code, valid_window=1):
            _log.warning(f"[login] invalid 2FA email={email}")
            _record_fail(ip)
            return RedirectResponse(url="/login?error=invalid_2fa&need_2fa=1", status_code=303)

    _clear_fails(ip)
    _log.info(f"[login] success user={user.id} email={email} trusted={device_trusted}")
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
    return templates.TemplateResponse("how_it_works.html", {"request": request, "user": user})


# --- Trade History ---

@app.get("/trade-history", response_class=HTMLResponse)
async def trade_history_page(request: Request):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    portfolio = _load_portfolio(user.id)
    all_trades = portfolio.get("trade_history", [])

    # Classify trades by vertical
    trades = []
    for t in reversed(all_trades):
        strategy = _s(t.get("strategy_id"))
        if strategy == "altcoin_hunter" or "Altcoin" in _s(t.get("reason")):
            trade_type = "crypto"
        elif strategy == "autopilot" or _s(t.get("symbol")).endswith("/USD"):
            trade_type = "stocks"
        elif strategy == "predictions" or _s(t.get("symbol")).startswith("PRED:"):
            trade_type = "predictions"
        else:
            trade_type = "crypto"  # default

        entry_time = t.get("entry_time", 0)
        exit_time = t.get("exit_time", 0)
        duration_s = exit_time - entry_time if exit_time > entry_time else 0
        if duration_s > 86400:
            duration_str = f"{duration_s / 86400:.1f}d"
        elif duration_s > 3600:
            duration_str = f"{duration_s / 3600:.1f}h"
        else:
            duration_str = f"{duration_s / 60:.0f}m"

        cost_usd = t.get("cost_usd") or t.get("amount_usd") or 0
        pnl_usd = round(t.get("pnl_usd") or 0, 2)

        # Reconstruct cost_usd from amount * entry_price if missing
        if not cost_usd and t.get("amount") and t.get("entry_price"):
            cost_usd = round(t["amount"] * t["entry_price"], 2)

        exit_amount = round(cost_usd + pnl_usd, 2) if cost_usd else round(abs(pnl_usd), 2)

        # Fee calculation: use stored fees or estimate at 0.6% per side
        fee_rate = 0.006
        entry_fee = t.get("entry_fee") or round(cost_usd * fee_rate, 2)
        exit_fee = t.get("exit_fee") or (round(exit_amount * fee_rate, 2) if exit_amount > 0 else 0)
        total_fees = t.get("total_fees") or round(entry_fee + exit_fee, 2)

        # Use stored net_pnl_usd ONLY if it's meaningful (non-zero when pnl_usd is non-zero)
        stored_net = t.get("net_pnl_usd")
        if stored_net is not None and stored_net != 0:
            net_pnl_usd = stored_net
        elif stored_net == 0 and pnl_usd == 0:
            net_pnl_usd = 0
        else:
            net_pnl_usd = round(pnl_usd - total_fees, 2)

        stored_net_pct = t.get("net_pnl_pct")
        if stored_net_pct is not None and stored_net_pct != 0:
            net_pnl_pct = stored_net_pct
        elif cost_usd > 0:
            net_pnl_pct = round(net_pnl_usd / cost_usd * 100, 2)
        else:
            # Fallback: use pnl_pct if available
            net_pnl_pct = round(t.get("pnl_pct") or 0, 2)

        trades.append({
            "symbol": t.get("symbol", ""),
            "type": trade_type,
            "side": t.get("side", "buy"),
            "entry_price": t.get("entry_price", 0),
            "exit_price": t.get("exit_price", 0),
            "cost_usd": round(cost_usd, 2),
            "exit_amount": exit_amount,
            "pnl_usd": pnl_usd,
            "pnl_pct": round(t.get("pnl_pct") or 0, 2),
            "total_fees": total_fees,
            "net_pnl_usd": net_pnl_usd,
            "net_pnl_pct": net_pnl_pct,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "exit_date": exit_time,
            "duration": duration_str,
            "reason": t.get("reason", ""),
        })

    # Build chart data: cumulative win rate over time per vertical
    # Use pnl_usd (gross P&L) as the primary win indicator — always reliably set
    def build_win_rate_series(trade_list):
        wins = 0
        points = []
        for i, t in enumerate(trade_list):
            if t.get("pnl_usd", 0) > 0:
                wins += 1
            rate = round(wins / (i + 1) * 100, 1)
            points.append({"x": t["exit_date"], "y": rate})
        return points

    # Trades are in reverse order (newest first), but chart needs oldest first
    trades_chrono = list(reversed(trades))
    crypto_trades = [t for t in trades_chrono if t["type"] == "crypto"]
    stocks_trades = [t for t in trades_chrono if t["type"] == "stocks"]
    pred_trades = [t for t in trades_chrono if t["type"] == "predictions"]

    chart_data = {
        "overall": build_win_rate_series(trades_chrono),
        "crypto": build_win_rate_series(crypto_trades),
        "stocks": build_win_rate_series(stocks_trades),
        "predictions": build_win_rate_series(pred_trades),
    }

    # KPI stats — use gross pnl_usd (always reliable) for win/loss determination
    def calc_kpi(trade_list):
        total = len(trade_list)
        wins = sum(1 for t in trade_list if t.get("pnl_usd", 0) > 0)
        return {"total": total, "wins": wins, "rate": round(wins / total * 100, 1) if total > 0 else 0}

    kpis = {
        "overall": calc_kpi(trades),
        "crypto": calc_kpi([t for t in trades if t["type"] == "crypto"]),
        "stocks": calc_kpi([t for t in trades if t["type"] == "stocks"]),
        "predictions": calc_kpi([t for t in trades if t["type"] == "predictions"]),
    }

    return templates.TemplateResponse("trade_history.html", {
        "request": request,
        "user": user,
        "trades": trades,
        "chart_data": chart_data,
        "kpis": kpis,
    })


# --- Onboarding Wizard ---

@app.get("/onboarding", response_class=HTMLResponse)
async def onboarding_page(request: Request):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    if user.onboarding_complete:
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse("onboarding.html", {
        "request": request, "user": user,
        "has_coinbase": user.has_coinbase,
        "has_alpaca": user.has_alpaca,
        "has_kalshi": user.has_kalshi,
        "has_perplexity": user.has_perplexity,
    })


@app.post("/onboarding/complete")
async def onboarding_complete(request: Request):
    """Mark onboarding as done and redirect to dashboard."""
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    from vesper.dashboard.database import set_onboarding_complete
    set_onboarding_complete(user.id)
    return RedirectResponse(url="/dashboard", status_code=303)


# --- Dashboard ---

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = _get_user(request)
    if not user:
        _log.debug("[dashboard] unauthenticated access, redirecting to login")
        return RedirectResponse(url="/login", status_code=303)

    # Redirect to onboarding wizard if not completed
    # Skip for existing users who already have data (pre-onboarding feature)
    if not user.onboarding_complete:
        portfolio = _load_portfolio(user.id)
        has_activity = (
            portfolio.get("positions") or portfolio.get("trade_history") or
            portfolio.get("altcoin_hunter", {}).get("enabled") or
            portfolio.get("autopilot", {}).get("enabled") or
            user.has_coinbase or user.has_alpaca or user.has_kalshi
        )
        if has_activity:
            from vesper.dashboard.database import set_onboarding_complete
            set_onboarding_complete(user.id)
        else:
            return RedirectResponse(url="/onboarding", status_code=303)

    portfolio = _load_portfolio(user.id)
    _log.info(f"[dashboard] user={user.id} email={user.email} mode={user.trading_mode}")
    cash = portfolio.get("cash", user.paper_balance)
    initial = portfolio.get("initial_balance", user.paper_balance)
    all_positions = portfolio.get("positions", {})
    all_trades = portfolio.get("trade_history", [])
    _log.info(f"[dashboard] positions={len(all_positions)} trades={len(all_trades)} cash={cash}")

    # Total portfolio value = cash + all autopilot funds (which include deployed positions)
    autopilot_funds = sum(
        portfolio.get(key, {}).get("fund_usd", 0)
        for key in ("altcoin_hunter", "autopilot", "predictions_autopilot")
    )
    portfolio_value = cash + autopilot_funds

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

    fmt_trades = []
    for t in reversed(trades[-50:]):
        fmt_trades.append({
            "symbol": t.get("symbol", ""), "side": t.get("side", ""),
            "entry_price": t.get("entry_price", 0), "exit_price": t.get("exit_price", 0),
            "pnl_usd": t.get("pnl_usd", 0), "pnl_pct": t.get("pnl_pct", 0),
            "entry_time": t.get("entry_time", 0),
            "exit_time": t.get("exit_time", 0),
            "reason": t.get("reason", ""),
        })

    fmt_pos = []
    for pid, p in positions.items():
        entry = p.get("entry_price") or 0
        amount = p.get("amount") or 0
        side = p.get("side", "buy")
        sl_price = (p.get("limits") or {}).get("stop_loss_price") or 0
        tp_max_price = (p.get("limits") or {}).get("take_profit_max_price") or 0
        cost = p.get("cost_usd") or 0
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
            "entry_time": p.get("entry_time", 0),
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
        "portfolio_value": portfolio_value,
        "initial_balance": initial, "total_pnl": total_pnl,
        "realized_pnl": realized_pnl,
        "total_pnl_pct": (total_pnl / initial * 100) if initial > 0 else 0,
        "win_rate": win_rate, "total_trades": len(trades),
        "wins": len(wins), "losses": len(losses),
        "positions": fmt_pos, "trades": fmt_trades,
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
async def save_keys(request: Request, api_key: str = Form(...), api_secret: str = Form(...),
                    redirect: str = Form("")):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    update_api_keys(user.id, api_key, api_secret)
    if redirect and redirect.startswith("/onboarding"):
        return RedirectResponse(url=redirect, status_code=303)
    return RedirectResponse(url="/settings?msg=keys_saved", status_code=303)

@app.post("/settings/api-keys/remove")
async def remove_keys(request: Request):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    update_api_keys(user.id, "", "")
    return RedirectResponse(url="/settings?msg=keys_saved", status_code=303)

@app.post("/settings/alpaca-keys")
async def save_alpaca_keys(request: Request, alpaca_key: str = Form(...), alpaca_secret: str = Form(...),
                           redirect: str = Form("")):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    update_alpaca_keys(user.id, alpaca_key, alpaca_secret)
    if redirect and redirect.startswith("/onboarding"):
        return RedirectResponse(url=redirect, status_code=303)
    return RedirectResponse(url="/settings?msg=keys_saved", status_code=303)

@app.post("/settings/alpaca-keys/remove")
async def remove_alpaca_keys(request: Request):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    update_alpaca_keys(user.id, "", "")
    return RedirectResponse(url="/settings?msg=keys_saved", status_code=303)

@app.post("/settings/kalshi-keys")
async def save_kalshi_keys(request: Request, kalshi_key: str = Form(...), kalshi_secret: str = Form(...),
                           redirect: str = Form("")):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    update_kalshi_keys(user.id, kalshi_key, kalshi_secret)
    if redirect and redirect.startswith("/onboarding"):
        return RedirectResponse(url=redirect, status_code=303)
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

@app.post("/settings/change-password")
async def change_password(request: Request,
                          current_password: str = Form(...),
                          new_password: str = Form(...),
                          confirm_password: str = Form(...)):
    import logging
    _log = logging.getLogger("vesper.dashboard")
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    # Verify current password
    if not verify_password(user, current_password):
        _log.warning(f"[change_password] user={user.id} — wrong current password")
        return RedirectResponse(url="/settings?msg=wrong_password", status_code=303)

    # Check passwords match
    if new_password != confirm_password:
        return RedirectResponse(url="/settings?msg=password_mismatch", status_code=303)

    # Validate password complexity
    errors = []
    if len(new_password) < 8:
        errors.append("at least 8 characters")
    if not any(c.isupper() for c in new_password):
        errors.append("one uppercase letter")
    if not any(c.islower() for c in new_password):
        errors.append("one lowercase letter")
    if not any(c.isdigit() for c in new_password):
        errors.append("one number")
    if not any(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?" for c in new_password):
        errors.append("one special character")
    if errors:
        return RedirectResponse(url="/settings?msg=password_weak", status_code=303)

    # Hash and update
    new_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    update_password(user.id, new_hash)
    _log.info(f"[change_password] user={user.id} — password changed successfully")
    return RedirectResponse(url="/settings?msg=password_changed", status_code=303)

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


@app.post("/settings/reset-trades")
async def reset_trades(request: Request):
    """Reset all trades, positions, and autopilot configs. Starts fresh."""
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    _log.warning(f"[reset-trades] user={user.id} email={user.email} — FULL RESET")

    portfolio_path = os.path.join(DATA_DIR, f"portfolio_{user.id}.json")
    fresh = {
        "cash": user.paper_balance,
        "initial_balance": user.paper_balance,
        "positions": {},
        "trade_history": [],
        "autopilot_log": [],
        "altcoin_hunter": {},
        "autopilot": {},
        "predictions_autopilot": {},
    }
    with open(portfolio_path, "w") as f:
        json.dump(fresh, f, indent=2)

    _log.info(f"[reset-trades] user={user.id} — portfolio reset to ${user.paper_balance}")
    return RedirectResponse(url="/settings?msg=trades_reset", status_code=303)


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
        r["time_fmt"] = r["created_at"]  # raw timestamp — formatted client-side

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
            "joined": u.created_at,  # raw timestamp — formatted client-side
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
    from vesper.ai_research import _get_platform_keys
    _pplx, _anth = _get_platform_keys()
    ai_keys = {
        "perplexity": bool(_pplx),
        "anthropic": bool(_anth),
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
    """Public (no auth) exchange for price data. Uses the auto-selected exchange."""
    try:
        from vesper.main import _resolve_exchange
        ex, _ = _resolve_exchange()
        return ex
    except Exception as e:
        _log.warning(f"[exchange] Failed to resolve public exchange: {e}")
        return None


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
    if not ex or not getattr(ex, 'markets', None):
        return result
    # Filter to symbols available on the selected exchange
    available_syms = [s for s in TICKER_SYMBOLS if s in ex.markets]
    try:
        tickers = ex.fetch_tickers(available_syms)
        for sym in available_syms:
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
        for sym in available_syms:
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
    if not ex:
        return []
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


# Symbol remaps for rebranded tokens (old pair -> new pair on exchanges)
_SYMBOL_REMAP = {
    "MATIC/USDT": "POL/USDT",
    "MATIC/USD": "POL/USD",
    "LUNC/USDT": "LUNA/USDT",
}


def _fetch_single_price(symbol: str) -> float:
    """Fetch a single crypto symbol's price — tries exchange, then CoinGecko fallback."""
    if not symbol:
        return 0
    ex = _get_public_exchange()

    # Try original symbol first, then remapped symbol
    symbols_to_try = [symbol]
    remapped = _SYMBOL_REMAP.get(symbol)
    if remapped:
        symbols_to_try.append(remapped)
    # Also try /USD if /USDT fails and vice versa
    if symbol.endswith("/USDT"):
        symbols_to_try.append(symbol.replace("/USDT", "/USD"))
    elif symbol.endswith("/USD") and not symbol.endswith("/BUSD"):
        symbols_to_try.append(symbol.replace("/USD", "/USDT"))

    for sym in symbols_to_try:
        try:
            if ex and hasattr(ex, 'markets') and ex.markets and sym in ex.markets:
                t = ex.fetch_ticker(sym)
                price = (t.get("last") or 0) if t else 0
                if price > 0:
                    return price
        except Exception as e:
            _log.debug(f"[price] exchange fetch failed {sym}: {e}")

    # Fallback: CoinGecko price cache (refreshed with market caps every 5min)
    base = symbol.split("/")[0].upper()
    if _coingecko_price_cache and base in _coingecko_price_cache:
        cg_price = _coingecko_price_cache[base]
        if cg_price > 0:
            _log.info(f"[price] CoinGecko fallback for {symbol}: ${cg_price}")
            return cg_price

    # Last resort: direct CoinGecko API call for this specific coin
    try:
        from vesper.sentiment import _COINGECKO_MAP
        coin_id = _COINGECKO_MAP.get(base)
        if coin_id:
            import urllib.request
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
            req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "Vesper/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
            price = data.get(coin_id, {}).get("usd", 0)
            if price > 0:
                _log.info(f"[price] CoinGecko direct for {symbol}: ${price}")
                return float(price)
    except Exception as e:
        _log.debug(f"[price] CoinGecko direct failed for {symbol}: {e}")

    _log.warning(f"[price] all sources failed for {symbol}")
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
        return (t.get("last") or 0) if t else 0
    except Exception:
        return 0


_position_price_cache: dict[str, tuple[float, float]] = {}  # symbol -> (price, timestamp)
_POSITION_PRICE_TTL = 15  # seconds

async def _get_price_map_for_positions(position_symbols: set[str], user=None) -> dict[str, float]:
    """Build a complete price map covering all position symbols.

    Multi-source approach:
    1. Cached exchange tickers (TICKER_SYMBOLS)
    2. Position price cache (15s TTL)
    3. CoinGecko price cache (populated with market caps, 5min TTL)
    4. Direct exchange fetch per symbol (with symbol remapping)
    5. Direct CoinGecko /simple/price API
    """
    position_symbols = {s for s in position_symbols if s}  # filter empty strings
    prices = await _get_cached_prices()
    price_map = {p["symbol"]: p["price"] for p in prices if p.get("price", 0) > 0}

    # Check position price cache for recently fetched prices
    now = time.time()
    missing = position_symbols - set(price_map.keys())
    still_missing = set()
    for sym in missing:
        cached = _position_price_cache.get(sym)
        if cached and (now - cached[1]) < _POSITION_PRICE_TTL and cached[0] > 0:
            price_map[sym] = cached[0]
        else:
            still_missing.add(sym)

    # Try CoinGecko price cache before making individual exchange calls
    if still_missing and _coingecko_price_cache:
        resolved = set()
        for sym in still_missing:
            base = sym.split("/")[0].upper()
            cg_price = _coingecko_price_cache.get(base, 0)
            if cg_price > 0:
                price_map[sym] = cg_price
                _position_price_cache[sym] = (cg_price, now)
                resolved.add(sym)
        still_missing -= resolved

    # Fetch remaining missing symbols individually (exchange + CoinGecko fallback)
    if still_missing:
        loop = asyncio.get_event_loop()
        for sym in still_missing:
            if sym.endswith("/USD"):
                price = await loop.run_in_executor(None, _fetch_stock_price, sym, user)
            else:
                price = await loop.run_in_executor(None, _fetch_single_price, sym)
            if price and price > 0:
                price_map[sym] = price
                _position_price_cache[sym] = (price, now)

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
    from vesper.ai_research import _get_platform_keys
    from vesper.state import cycle_state as _cycle_state
    pplx, anth = _get_platform_keys()
    return {
        "status": "running",
        "time": datetime.now().isoformat(),
        "ai_keys": {
            "perplexity": bool(pplx),
            "anthropic": bool(anth),
        },
        "cycles": _cycle_state,
    }


@app.post("/api/admin/deploy")
async def admin_deploy():
    """Hot-deploy: git pull inside container + restart process.
    Docker restart policy brings us back with new code.
    For full rebuild (Dockerfile changes), use SSH deploy."""
    import subprocess
    branch = "claude/deploy-openclaw-cloudflare-GkBQL"
    app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    try:
        result = subprocess.run(
            ["git", "pull", "origin", branch],
            capture_output=True, text=True, timeout=60, cwd=app_dir,
        )
        pull_output = (result.stdout + result.stderr).strip()
        if result.returncode != 0:
            return {"status": "error", "phase": "git_pull", "output": pull_output}

        if "Already up to date" in pull_output:
            return {"status": "already_current", "output": pull_output}

        # Restart process — Docker restart=unless-stopped brings us back with new code
        import threading
        def _restart():
            import time as _t
            _t.sleep(1)
            os._exit(0)
        threading.Thread(target=_restart, daemon=True).start()

        return {"status": "deploying", "output": pull_output,
                "message": "Code pulled. Restarting in 1s..."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/admin/bot-state")
async def bot_state():
    """Debug endpoint — show all active users and their autopilot configs."""
    from vesper.dashboard.database import get_active_users, get_all_users, init_db
    init_db()
    all_users = get_all_users()
    active_users = get_active_users()
    result = {
        "total_users": len(all_users),
        "active_users": len(active_users),
        "users": [],
    }
    for u in all_users:
        p = _load_portfolio(u.id)
        user_info = {
            "id": u.id,
            "email": u.email,
            "bot_active": u.bot_active,
            "trading_mode": u.trading_mode,
            "altcoin_hunter": p.get("altcoin_hunter", {}),
            "autopilot": p.get("autopilot", {}),
            "predictions_autopilot": p.get("predictions_autopilot", {}),
            "positions_count": len(p.get("positions", {})),
            "trade_history_count": len(p.get("trade_history", [])),
            "autopilot_log_count": len(p.get("autopilot_log", [])),
            "last_logs": p.get("autopilot_log", [])[-5:],
        }
        result["users"].append(user_info)
    return result


@app.get("/api/learning-state")
async def api_learning_state(request: Request):
    """Get the current learning engine state and strategy performance profiles."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    portfolio = _load_portfolio(user.id)
    learning = portfolio.get("learning_state", {})
    if not learning:
        return {"status": "not_initialized", "message": "Learning engine activates after 8+ closed trades."}
    return learning


@app.get("/api/admin/scan-test")
async def scan_test():
    """Diagnostic endpoint — test crypto exchanges + prediction markets.

    Tests which crypto exchanges are reachable from this server,
    then verifies full scanning pipeline (OHLCV + indicators + scoring).
    """
    import traceback
    import ccxt as _ccxt
    from vesper.market_data import get_market_snapshot, get_multi_tf_snapshot
    from vesper.strategies.altcoin_hunter import compute_trend_score
    results = {}

    # Test which crypto exchanges work from this server
    exchanges_to_test = [
        ("bybit", lambda: _ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "spot"}})),
        ("binanceus", lambda: _ccxt.binanceus({"enableRateLimit": True})),
        ("kucoin", lambda: _ccxt.kucoin({"enableRateLimit": True})),
        ("okx", lambda: _ccxt.okx({"enableRateLimit": True})),
        ("binance", lambda: _ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})),
    ]
    working_exchange = None
    exchange_results = {}
    for name, factory in exchanges_to_test:
        try:
            ex = factory()
            ex.load_markets()
            ohlcv = ex.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=5)
            price = ohlcv[-1][4] if ohlcv else 0
            exchange_results[name] = {"ok": True, "price": round(price, 2), "markets": len(ex.markets)}
            if working_exchange is None:
                working_exchange = ex
                results["selected_exchange"] = name
        except Exception as e:
            exchange_results[name] = {"ok": False, "error": str(e)[:200]}
    results["exchanges"] = exchange_results

    if not working_exchange:
        results["error"] = "No crypto exchange reachable from this server"
        return results

    # Test full pipeline with working exchange
    try:
        snap = get_market_snapshot(working_exchange, "BTC/USDT", timeframe="1h")
        results["btc_snapshot"] = {
            "ok": True,
            "price": round(snap["price"], 2),
            "rsi": round(snap.get("rsi", 0), 2),
            "adx": round(snap.get("adx", 0), 2),
        }
    except Exception as e:
        results["btc_snapshot"] = {"ok": False, "error": str(e)[:300]}

    try:
        snap = get_multi_tf_snapshot(working_exchange, "ETH/USDT")
        score = compute_trend_score(snap)
        results["eth_score"] = {
            "ok": True,
            "score": score["score"],
            "signal": score["signal"].name,
            "factors": score["factors"],
        }
    except Exception as e:
        results["eth_score"] = {"ok": False, "error": str(e)[:300]}

    # Test parallel scan with thread-local exchanges
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        test_coins = ["SOL/USDT", "XRP/USDT", "DOGE/USDT"]

        selected_name = results["selected_exchange"]
        selected_factory = dict(exchanges_to_test)[selected_name]
        cached_markets = working_exchange.markets
        cached_markets_by_id = working_exchange.markets_by_id
        cached_currencies = working_exchange.currencies
        cached_currencies_by_id = working_exchange.currencies_by_id

        def _test_scan(sym):
            tex = selected_factory()
            tex.markets = cached_markets
            tex.markets_by_id = cached_markets_by_id
            tex.currencies = cached_currencies
            tex.currencies_by_id = cached_currencies_by_id
            s = get_market_snapshot(tex, sym, timeframe="1h")
            sc = compute_trend_score(s)
            return {"symbol": sym, "price": round(s["price"], 4), "score": sc["score"], "signal": sc["signal"].name}

        thread_results = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futs = {pool.submit(_test_scan, c): c for c in test_coins}
            for f in as_completed(futs):
                thread_results.append(f.result())
        results["parallel_scan"] = {"ok": True, "results": thread_results}
    except Exception as e:
        results["parallel_scan"] = {"ok": False, "error": str(e)[:300]}

    # Test prediction markets
    try:
        from vesper.polymarket import get_trending_markets
        poly = get_trending_markets(limit=5, max_days=14)
        results["polymarket"] = {"ok": True, "count": len(poly)}
    except Exception as e:
        results["polymarket"] = {"ok": False, "error": str(e)[:200]}

    try:
        from vesper.kalshi import get_kalshi_markets
        kalshi = get_kalshi_markets(limit=5, max_days=14)
        results["kalshi"] = {"ok": True, "count": len(kalshi)}
    except Exception as e:
        results["kalshi"] = {"ok": False, "error": str(e)[:200]}

    return results


@app.post("/api/admin/api-keys")
async def update_api_keys(request: Request):
    """Admin endpoint — update platform LLM API keys (persisted to data volume).

    Auth: admin session OR existing valid Perplexity key in X-Admin-Key header.
    """
    user = _get_user(request)
    admin_key = request.headers.get("X-Admin-Key", "")
    from vesper.ai_research import _get_platform_keys
    pplx, _ = _get_platform_keys()
    if not (user and user.is_admin) and admin_key != pplx:
        return {"error": "admin only"}

    body = await request.json()
    keys_file = os.path.join(os.environ.get("VESPER_DATA_DIR", "data"), "api_keys.json")

    # Load existing keys
    existing = {}
    try:
        with open(keys_file) as f:
            existing = json.load(f)
    except (FileNotFoundError, ValueError):
        pass

    # Update only provided keys
    for k in ("PERPLEXITY_API_KEY", "ANTHROPIC_API_KEY"):
        if k in body and body[k]:
            existing[k] = body[k]

    with open(keys_file, "w") as f:
        json.dump(existing, f)

    return {"ok": True, "keys_updated": list(body.keys())}


@app.get("/api/admin/diagnostics")
async def admin_diagnostics(request: Request):
    """Debug endpoint — check API usage DB, test LLM keys."""

    import sqlite3

    # Check api_usage table
    db_path = os.path.join(
        os.environ.get("VESPER_DATA_DIR", "data"), "vesper.db"
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
    # Use the same key resolution as ai_research (env + file fallback)
    from vesper.ai_research import _get_platform_keys
    pplx_key, anth_key = _get_platform_keys()

    pplx_ok = None
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
            _log.warning("[ws] auth failed — invalid session")
            await ws.send_json({"type": "error", "msg": "unauthorized"})
            return
        user = get_user_by_id(user_id)
        if not user:
            _log.warning(f"[ws] auth failed — user_id={user_id} not found")
            await ws.send_json({"type": "error", "msg": "unauthorized"})
            return
        _log.info(f"[ws] connected user={user.id}")

        await ws.send_json({"type": "auth_ok"})

        # Push loop: send fresh data every 2 seconds
        while True:
            try:
                portfolio = _load_portfolio(user.id)
                all_positions = portfolio.get("positions", {})

                # --- Positions (send ALL with trade_mode, client filters) ---
                positions = all_positions
                pos_list = []
                if positions:
                    pos_symbols = {
                        _s(p.get("symbol")) for p in positions.values()
                        if not _s(p.get("symbol")).startswith("PRED:")
                    }
                    price_map = await _get_price_map_for_positions(pos_symbols, user)

                    for pid, p in positions.items():
                        sym = _s(p.get("symbol"))
                        entry = p.get("entry_price") or 0
                        if sym.startswith("PRED:"):
                            current = p.get("current_probability") or entry
                        else:
                            current = price_map.get(sym, entry) or 0
                        amount = p.get("amount") or 0
                        side = p.get("side", "buy")
                        pnl_usd = (current - entry) * amount if side == "buy" else (entry - current) * amount
                        pnl_pct = ((current - entry) / entry * 100) if entry > 0 and side == "buy" else (
                            ((entry - current) / entry * 100) if entry > 0 else 0)

                        limits = p.get("limits") or {}
                        sl = limits.get("stop_loss_price") or 0
                        tp_max = limits.get("take_profit_max_price") or 0
                        price_range = tp_max - sl if tp_max > sl else 1
                        progress = max(0, min(100, ((current - sl) / price_range) * 100))

                        cost_usd = p.get("cost_usd") or 0
                        tp_min = limits.get("take_profit_min_price") or 0
                        if side == "buy":
                            max_loss = abs((entry - sl) * amount) if sl > 0 else cost_usd
                            max_win = abs((tp_max - entry) * amount) if tp_max > 0 else 0
                        else:
                            max_loss = abs((sl - entry) * amount) if sl > 0 else cost_usd
                            max_win = abs((entry - tp_max) * amount) if tp_max > 0 else 0

                        trailing_pct = p.get("trailing_stop_pct") or 0
                        highest_seen = p.get("highest_price_seen") or 0
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
                autopilot_funds = sum(
                    portfolio.get(k, {}).get("fund_usd", 0)
                    for k in ("altcoin_hunter", "autopilot", "predictions_autopilot")
                )

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
                        "portfolio_value": round(cash + autopilot_funds + m_unrealized, 2),
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
    try:
        return await _portfolio_stats_inner(user, mode)
    except Exception:
        _log.error(f"[portfolio-stats] user={user.id} error:\n{traceback.format_exc()}")
        return JSONResponse({"error": "internal error"}, status_code=500)

async def _portfolio_stats_inner(user, mode: str):
    portfolio = _load_portfolio(user.id)
    _log.info(f"[portfolio-stats] user={user.id} mode={mode or user.trading_mode}")
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
        pos_symbols = {_s(p.get("symbol")) for p in positions.values() if not _s(p.get("symbol")).startswith("PRED:")}
        price_map = await _get_price_map_for_positions(pos_symbols, user)
        for p in positions.values():
            entry = p.get("entry_price") or 0
            sym = _s(p.get("symbol"))
            if sym.startswith("PRED:"):
                current = p.get("current_probability") or entry
            else:
                current = price_map.get(sym, entry) or 0
            amount = p.get("amount") or 0
            side = p.get("side", "buy")
            if side == "buy":
                unrealized_pnl += (current - entry) * amount
            else:
                unrealized_pnl += (entry - current) * amount

    total_pnl = realized_pnl + unrealized_pnl
    autopilot_funds = sum(
        (portfolio.get(k) or {}).get("fund_usd") or 0
        for k in ("altcoin_hunter", "autopilot", "predictions_autopilot")
    )
    portfolio_value = cash + autopilot_funds + unrealized_pnl

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


@app.get("/api/portfolio-history")
async def api_portfolio_history(request: Request, hours: int = 24):
    """Return portfolio value snapshots for the dashboard chart.

    Snapshots are recorded every ~1 min by the bot. Supports up to 6 months.
    Returns downsampled data: max ~500 points for performance.
    """
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    portfolio = _load_portfolio(user.id)
    snapshots = portfolio.get("portfolio_snapshots", [])

    # Filter by time range
    cutoff = time.time() - (hours * 3600)
    filtered = [s for s in snapshots if s.get("t", 0) > cutoff]

    # Downsample if too many points (keep max ~500 for chart performance)
    if len(filtered) > 500:
        step = len(filtered) // 500
        filtered = filtered[::step] + [filtered[-1]]

    return {"snapshots": filtered}


@app.get("/api/positions")
async def api_positions(request: Request, mode: str = ""):
    """Open positions with live unrealized P&L."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        return await _positions_inner(user, mode)
    except Exception:
        _log.error(f"[positions] user={user.id} error:\n{traceback.format_exc()}")
        return JSONResponse({"error": "internal error"}, status_code=500)

async def _positions_inner(user, mode: str = ""):
    portfolio = _load_portfolio(user.id)
    all_positions = portfolio.get("positions", {})
    # Filter by trade_mode
    if mode in ("paper", "real"):
        current_mode = mode
    else:
        mode_match = {"live": "real", "paper": "paper"}
        current_mode = mode_match.get(user.trading_mode, "paper")
    positions = {
        pid: p for pid, p in all_positions.items()
        if p.get("trade_mode", "paper") == current_mode
    }
    _log.info(f"[positions] user={user.id} count={len(positions)}")

    if not positions:
        return []

    # Build price map that covers ALL position symbols (not just TICKER_SYMBOLS)
    pos_symbols = {_s(p.get("symbol")) for p in positions.values() if not _s(p.get("symbol")).startswith("PRED:")}
    price_map = await _get_price_map_for_positions(pos_symbols, user)

    result = []
    for pid, p in positions.items():
        sym = _s(p.get("symbol"))
        entry = p.get("entry_price") or 0
        # For prediction positions (PRED:*), use stored probability data for P&L
        is_prediction = sym.startswith("PRED:")
        if is_prediction:
            stored_prob = p.get("current_probability") or entry
            current = stored_prob
        else:
            current = price_map.get(sym, entry) or 0
        amount = p.get("amount") or 0
        side = p.get("side", "buy")

        if side == "buy":
            pnl_usd = (current - entry) * amount
            pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
        else:
            pnl_usd = (entry - current) * amount
            pnl_pct = ((entry - current) / entry * 100) if entry > 0 else 0

        limits = p.get("limits") or {}
        sl = limits.get("stop_loss_price") or 0
        tp_max = limits.get("take_profit_max_price") or 0
        # Progress: 0% at stop-loss, 100% at take-profit
        price_range = tp_max - sl if tp_max > sl else 1
        progress = max(0, min(100, ((current - sl) / price_range) * 100))

        cost_usd = p.get("cost_usd") or 0
        tp_min = limits.get("take_profit_min_price") or 0
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
        trailing_sl = 0
        trailing_pct = p.get("trailing_stop_pct") or 0
        highest_seen = p.get("highest_price_seen") or 0
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
            pdata = _clean_nones(json.load(f))
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

    # Store analysis data for "Learn More" modal
    analyses = pdata.get("position_analyses", {})
    analyses[pos_id] = {
        "deep_research": {
            "signal": "BUY",
            "confidence": 0,
            "reasoning": f"Manual trade: {strategy_id}. Entry at ${current_price:,.2f}, "
                         f"SL {stop_loss_pct}%, TP {tp_min_pct}-{tp_max_pct}%"
                         + (f", trailing {trailing_stop_pct}%" if trailing_stop_pct > 0 else ""),
        },
        "indicators": {},
        "risk_level": "moderate" if stop_loss_pct <= 3 else "aggressive",
    }
    # Prune old analyses
    if len(analyses) > 50:
        keys = sorted(analyses.keys())
        for k in keys[:len(analyses) - 50]:
            del analyses[k]
    pdata["position_analyses"] = analyses

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
    cost_usd = pos.get("cost_usd", 0)
    if pos.get("side", "buy") == "buy":
        pnl_usd = (current_price - entry) * amount
        pnl_pct = ((current_price - entry) / entry * 100) if entry > 0 else 0
    else:
        pnl_usd = (entry - current_price) * amount
        pnl_pct = ((entry - current_price) / entry * 100) if entry > 0 else 0

    # Calculate fees (0.6% taker fee per side for Coinbase/exchanges)
    fee_rate = 0.006
    entry_fee = pos.get("est_fee", round(cost_usd * fee_rate, 2))
    exit_value = current_price * amount
    exit_fee = round(exit_value * fee_rate, 2)
    total_fees = round(entry_fee + exit_fee, 2)
    net_pnl_usd = round(pnl_usd - total_fees, 2)
    net_pnl_pct = round((net_pnl_usd / cost_usd * 100) if cost_usd > 0 else 0, 2)

    # Record trade (preserve trade_mode for siloing)
    trade = {
        "symbol": pos["symbol"],
        "side": pos.get("side", "buy"),
        "entry_price": entry,
        "exit_price": current_price,
        "amount": amount,
        "cost_usd": cost_usd,
        "pnl_usd": round(pnl_usd, 2),
        "pnl_pct": round(pnl_pct, 2),
        "entry_fee": entry_fee,
        "exit_fee": exit_fee,
        "total_fees": total_fees,
        "net_pnl_usd": net_pnl_usd,
        "net_pnl_pct": net_pnl_pct,
        "entry_time": pos.get("entry_time", 0),
        "exit_time": time.time(),
        "reason": "Manual close",
        "strategy_reason": pos.get("strategy_reason", ""),
        "trade_mode": pos.get("trade_mode", "paper"),
    }

    pdata["cash"] = pdata.get("cash", 0) + cost_usd + pnl_usd
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
        "net_pnl_usd": net_pnl_usd,
        "total_fees": total_fees,
        "cost_usd": cost_usd,
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
async def api_autopilot_status(request: Request, mode: str = ""):
    """Get autopilot status and positions."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    portfolio = _load_portfolio(user.id)
    autopilot = portfolio.get("autopilot", {})
    all_positions = portfolio.get("positions", {})
    # Filter by trade_mode
    if mode in ("paper", "real"):
        current_mode = mode
    else:
        mode_match = {"live": "real", "paper": "paper"}
        current_mode = mode_match.get(user.trading_mode, "paper")
    positions = {
        pid: p for pid, p in all_positions.items()
        if p.get("trade_mode", "paper") == current_mode
    }

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
        and t.get("trade_mode", "paper") == current_mode
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
            pdata = _clean_nones(json.load(f))

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
async def api_altcoin_hunter_status(request: Request, mode: str = ""):
    """Get altcoin hunter status and positions."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        return await _altcoin_hunter_status_inner(user, mode)
    except Exception:
        _log.error(f"[altcoin-hunter] user={user.id} GET error:\n{traceback.format_exc()}")
        return JSONResponse({"error": "internal error"}, status_code=500)

async def _altcoin_hunter_status_inner(user, mode: str = ""):
    portfolio = _load_portfolio(user.id)
    _log.info(f"[altcoin-hunter] user={user.id} GET status mode={mode or user.trading_mode}")
    hunter = portfolio.get("altcoin_hunter", {})
    all_positions = portfolio.get("positions", {})
    # Filter by trade_mode
    if mode in ("paper", "real"):
        current_mode = mode
    else:
        mode_match = {"live": "real", "paper": "paper"}
        current_mode = mode_match.get(user.trading_mode, "paper")
    positions = {
        pid: p for pid, p in all_positions.items()
        if p.get("trade_mode", "paper") == current_mode
    }

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

    # Compute realized P&L from closed trades (filtered by mode)
    all_trades = portfolio.get("trade_history", [])
    realized_pnl = sum(
        t.get("pnl_usd", 0) for t in all_trades
        if t.get("trade_mode", "paper") == current_mode
        and (_s(t.get("strategy_id")) == "altcoin_hunter" or "Altcoin Hunter" in _s(t.get("reason")))
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
    try:
        body = await request.json()
        _log.info(f"[altcoin-hunter] user={user.id} POST action={body.get('action')} amount={body.get('amount_usd')}")
    except Exception:
        _log.error(f"[altcoin-hunter] user={user.id} POST bad JSON:\n{traceback.format_exc()}")
        return JSONResponse({"error": "invalid request body"}, status_code=400)
    action = body.get("action", "")
    amount = float(body.get("amount_usd") or 0)

    portfolio_path = os.path.join(
        os.environ.get("VESPER_DATA_DIR", "data"),
        f"portfolio_{user.id}.json",
    )
    pdata = {}
    if os.path.exists(portfolio_path):
        with open(portfolio_path) as f:
            pdata = _clean_nones(json.load(f))

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
async def api_predictions_autopilot_status(request: Request, mode: str = ""):
    """Get predictions autopilot status and positions."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    portfolio = _load_portfolio(user.id)
    pred_cfg = portfolio.get("predictions_autopilot", {})
    all_positions = portfolio.get("positions", {})
    # Filter by trade_mode
    if mode in ("paper", "real"):
        current_mode = mode
    else:
        mode_match = {"live": "real", "paper": "paper"}
        current_mode = mode_match.get(user.trading_mode, "paper")
    positions = {
        pid: p for pid, p in all_positions.items()
        if p.get("trade_mode", "paper") == current_mode
    }

    pred_positions = []
    deployed = 0.0
    total_unrealized = 0.0
    for pid, p in positions.items():
        if p.get("strategy_id") == "predictions":
            cost = p.get("cost_usd", 0)
            deployed += cost
            entry = p.get("entry_price", 0)
            current = p.get("current_probability") or entry
            amount = p.get("amount", 0)
            side = p.get("side", "buy")
            if side == "buy":
                pnl_usd = (current - entry) * amount
                pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
            else:
                pnl_usd = (entry - current) * amount
                pnl_pct = ((entry - current) / entry * 100) if entry > 0 else 0
            total_unrealized += pnl_usd
            pred_positions.append({
                "id": pid,
                "symbol": p.get("symbol", ""),
                "entry_price": entry,
                "current_price": current,
                "cost_usd": cost,
                "amount": amount,
                "pnl_usd": round(pnl_usd, 2),
                "pnl_pct": round(pnl_pct, 2),
                "entry_time": p.get("entry_time", 0),
                "strategy_reason": p.get("strategy_reason", ""),
                "prediction_side": p.get("prediction_side", ""),
                "prediction_question": p.get("prediction_question", ""),
                "prediction_ai_prob": p.get("prediction_ai_prob", 0),
                "prediction_mkt_prob": p.get("prediction_mkt_prob", 0),
                "prediction_edge": p.get("prediction_edge", 0),
            })

    fund_total = pred_cfg.get("fund_usd", 0)

    # Compute realized P&L from closed predictions (filtered by mode)
    all_trades = portfolio.get("trade_history", [])
    realized_pnl = sum(
        t.get("pnl_usd", 0) for t in all_trades
        if t.get("strategy_id") == "predictions"
        and t.get("trade_mode", "paper") == current_mode
    )

    # Get recent prediction scan logs
    all_logs = portfolio.get("autopilot_log", [])
    pred_logs = [l for l in all_logs if l.get("type") == "predictions_scan"][-10:]

    return {
        "enabled": pred_cfg.get("enabled", False),
        "fund_usd": fund_total,
        "deployed_usd": round(deployed, 2),
        "available_usd": round(fund_total - deployed, 2),
        "unrealized_pnl": round(total_unrealized, 2),
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
            pdata = _clean_nones(json.load(f))

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
        # Write an initial scan log so the UI shows immediate feedback
        log = pdata.get("autopilot_log", [])
        log.append({
            "type": "predictions_scan",
            "time": int(time.time()),
            "markets_scanned": 0,
            "status": "starting",
            "positions": 0,
            "max_positions": int(body.get("max_positions", 20)),
            "deployed_usd": 0,
            "available_usd": amount,
            "actions": [],
        })
        pdata["autopilot_log"] = log[-50:]
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
    pos = portfolio.get("positions", {}).get(pid, {})
    analyses = portfolio.get("position_analyses", {})
    analysis = analyses.get(pid, {})

    if analysis:
        # Flatten deep_research into top-level fields for the frontend
        deep = analysis.get("deep_research", {})
        result = {
            "found": True,
            "signal": deep.get("signal", "BUY" if pos.get("side", "buy") == "buy" else "SELL"),
            "confidence": deep.get("confidence", pos.get("confidence", 0)),
            "reasoning": deep.get("reasoning", ""),
            "search_summary": deep.get("search_summary", ""),
            "bullish_factors": deep.get("bullish_factors", []),
            "bearish_factors": deep.get("bearish_factors", []),
            "catalysts": deep.get("catalysts", []),
            "sources": deep.get("sources", []),
            "indicators": analysis.get("indicators", {}),
            "trend_score": analysis.get("trend_score"),
            "trend_factors": analysis.get("trend_factors"),
            "risk_level": analysis.get("risk_level", ""),
            "prediction_meta": analysis.get("prediction_meta"),
            "social_sentiment": analysis.get("social_sentiment"),
        }
        # Include position context
        if pos:
            result["entry_price"] = pos.get("entry_price", 0)
            result["cost_usd"] = pos.get("cost_usd", 0)
            result["strategy_id"] = pos.get("strategy_id", "")
            result["symbol"] = pos.get("symbol", "")
        return result

    # Fallback: no pre-saved analysis — run live deep research on-demand
    if not pos:
        return {"found": False, "reasoning": "Position not found."}

    reason = _s(pos.get("strategy_reason"))
    strategy_id = _s(pos.get("strategy_id"))
    symbol = _s(pos.get("symbol"))
    entry_price = pos.get("entry_price", 0)

    # Run live deep research for this position (Perplexity + Claude)
    live_research = {}
    social = {}
    is_prediction = symbol.startswith("PRED:")

    try:
        loop = asyncio.get_event_loop()
        if is_prediction:
            question = _s(pos.get("prediction_question")) or symbol.replace("PRED:", "")
            mkt_prob = pos.get("prediction_mkt_prob", 50)
            category = ""
            from vesper.ai_research import research_market
            live_research = await loop.run_in_executor(
                None, research_market, question, mkt_prob, "", category
            )
        else:
            from vesper.ai_research import research_asset
            # Get current market indicators if available
            try:
                ex = _get_public_exchange()
                if symbol.endswith("/USD"):
                    from vesper.market_data import get_stock_snapshot
                    snapshot = await loop.run_in_executor(None, get_stock_snapshot, symbol)
                else:
                    from vesper.market_data import get_multi_tf_snapshot
                    snapshot = await loop.run_in_executor(None, get_multi_tf_snapshot, ex, symbol)
            except Exception:
                snapshot = {"price": entry_price}

            asset_type = "stock" if symbol.endswith("/USD") else "crypto"
            live_research = await loop.run_in_executor(
                None, research_asset, symbol, {}, snapshot.get("price", entry_price), asset_type
            )

            # Also fetch social sentiment
            try:
                from vesper.social_sentiment import get_social_signal
                social = await loop.run_in_executor(None, get_social_signal, symbol, asset_type)
            except Exception:
                social = {}
    except Exception as e:
        _log.warning(f"[position-analysis] Live research failed for {symbol}: {e}")

    # Build rich analysis from live research
    deep = live_research if live_research.get("researched") else {}

    result = {
        "found": True,
        "signal": deep.get("signal", "BUY" if pos.get("side", "buy") == "buy" else "SELL"),
        "confidence": deep.get("confidence", 0),
        "reasoning": deep.get("reasoning", reason),
        "search_summary": deep.get("search_summary", ""),
        "bullish_factors": deep.get("bullish_factors", []),
        "bearish_factors": deep.get("bearish_factors", []),
        "catalysts": deep.get("catalysts", []),
        "sources": deep.get("sources", []),
        "entry_price": entry_price,
        "cost_usd": pos.get("cost_usd", 0),
        "strategy_id": strategy_id,
        "symbol": symbol,
        "risk_level": "",
    }

    if social:
        result["social_sentiment"] = social

    # Include prediction metadata
    if pos.get("prediction_question"):
        result["prediction_meta"] = {
            "question": pos.get("prediction_question", ""),
            "ai_prob": pos.get("prediction_ai_prob", 0),
            "mkt_prob": pos.get("prediction_mkt_prob", 0),
            "edge": pos.get("prediction_edge", 0),
            "side": pos.get("prediction_side", ""),
        }

    # Parse trend factors from strategy_reason for altcoin_hunter
    if strategy_id == "altcoin_hunter" and reason:
        import re
        score_m = re.search(r"score\s+([\d.]+)", reason)
        if score_m:
            result["trend_score"] = float(score_m.group(1))
        factors = {}
        for factor_name in ("momentum", "rsi", "ema_alignment", "volume_surge", "adx_strength", "btc_relative"):
            m = re.search(rf"{factor_name}=([\d.]+)", reason)
            if m:
                factors[factor_name] = float(m.group(1))
        if factors:
            result["trend_factors"] = factors

    # If live research didn't return factors, parse from strategy_reason text
    if not result["bullish_factors"] and not result["bearish_factors"] and reason:
        bullish = []
        bearish = []
        for part in reason.replace(";", ".").replace("|", ".").split("."):
            line = part.strip()
            if not line:
                continue
            lower = line.lower()
            if any(w in lower for w in ["bullish", "uptrend", "above", "support", "momentum", "surge", "accumulation", "buy", "positive", "strong"]):
                bullish.append(line)
            elif any(w in lower for w in ["bearish", "downtrend", "below", "resistance", "overbought", "sell", "negative", "weak"]):
                bearish.append(line)
        if bullish:
            result["bullish_factors"] = bullish[:5]
        if bearish:
            result["bearish_factors"] = bearish[:5]

    # Cache the analysis for future requests
    try:
        analysis_data = {
            "deep_research": {
                "signal": result.get("signal", ""),
                "confidence": result.get("confidence", 0),
                "reasoning": result.get("reasoning", ""),
                "bullish_factors": result.get("bullish_factors", []),
                "bearish_factors": result.get("bearish_factors", []),
                "catalysts": result.get("catalysts", []),
                "sources": result.get("sources", []),
                "search_summary": result.get("search_summary", ""),
            },
            "social_sentiment": social,
            "risk_level": result.get("risk_level", ""),
        }
        if result.get("trend_score"):
            analysis_data["trend_score"] = result["trend_score"]
        if result.get("trend_factors"):
            analysis_data["trend_factors"] = result["trend_factors"]
        if result.get("prediction_meta"):
            analysis_data["prediction_meta"] = result["prediction_meta"]
        analyses[pid] = analysis_data
        portfolio["position_analyses"] = analyses
        _save_portfolio(user.id, portfolio)
    except Exception:
        pass  # Non-critical — caching failure shouldn't break the response

    return result


@app.get("/api/social-sentiment/{symbol:path}")
async def api_social_sentiment(request: Request, symbol: str, asset_type: str = "crypto"):
    """Get X/Twitter social sentiment for a symbol."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    from vesper.social_sentiment import fetch_social_sentiment
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, fetch_social_sentiment, symbol, asset_type)
    if not result:
        return JSONResponse({"error": "Social sentiment unavailable — check Perplexity API key"}, status_code=503)
    return result


# ═══════════════════════════════════════
# Markets Browser (Crypto + Stocks)
# ═══════════════════════════════════════

_markets_crypto_cache: dict = {}
_markets_stocks_cache: dict = {}


_coingecko_mcap_cache: dict = {}
_coingecko_price_cache: dict[str, float] = {}  # symbol (e.g. "BTC") -> USD price
_coingecko_price_ts: float = 0


def _fetch_coingecko_market_caps() -> dict:
    """Fetch market caps + prices from CoinGecko free API (cached 5min)."""
    global _coingecko_mcap_cache, _coingecko_price_cache, _coingecko_price_ts
    now = time.time()
    if _coingecko_mcap_cache and now - _coingecko_mcap_cache.get("ts", 0) < 300:
        return _coingecko_mcap_cache.get("data", {})

    import urllib.request
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page=1&sparkline=false"
        req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "Vesper/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            coins = json.loads(resp.read())
        mcap_map = {}
        price_map = {}
        for c in coins:
            sym = (c.get("symbol") or "").upper()
            mcap_map[sym] = {
                "market_cap": c.get("market_cap") or 0,
                "name": c.get("name") or sym,
            }
            price = c.get("current_price")
            if price and price > 0:
                price_map[sym] = float(price)
        _coingecko_mcap_cache = {"data": mcap_map, "ts": now}
        _coingecko_price_cache = price_map
        _coingecko_price_ts = now
        return mcap_map
    except Exception:
        return _coingecko_mcap_cache.get("data", {})


def _fetch_crypto_markets_sync() -> list[dict]:
    """Fetch full ALTCOIN_UNIVERSE tickers with volume + market cap."""
    from config.settings import ALTCOIN_UNIVERSE
    ex = _get_public_exchange()
    if not ex or not getattr(ex, 'markets', None):
        return []
    available = [s for s in ALTCOIN_UNIVERSE if s in ex.markets]

    # Fetch market caps from CoinGecko
    mcap_data = _fetch_coingecko_market_caps()

    result = []
    try:
        tickers = ex.fetch_tickers(available)
        for sym in available:
            t = tickers.get(sym)
            if not t:
                continue
            base = sym.split("/")[0]
            cg = mcap_data.get(base, {})
            result.append({
                "symbol": sym,
                "name": cg.get("name", base),
                "price": t.get("last", 0) or 0,
                "change_pct": _calc_change_pct(t),
                "volume": (t.get("quoteVolume") or t.get("baseVolume", 0) or 0),
                "market_cap": cg.get("market_cap", 0),
                "high": t.get("high", 0) or 0,
                "low": t.get("low", 0) or 0,
            })
    except Exception:
        for sym in available[:20]:
            try:
                t = ex.fetch_ticker(sym)
                base = sym.split("/")[0]
                cg = mcap_data.get(base, {})
                result.append({
                    "symbol": sym,
                    "name": cg.get("name", base),
                    "price": t.get("last", 0) or 0,
                    "change_pct": _calc_change_pct(t),
                    "volume": (t.get("quoteVolume") or t.get("baseVolume", 0) or 0),
                    "market_cap": cg.get("market_cap", 0),
                    "high": t.get("high", 0) or 0,
                    "low": t.get("low", 0) or 0,
                })
            except Exception:
                pass
    # Sort by market cap descending (fallback to volume)
    result.sort(key=lambda x: x.get("market_cap", 0) or x.get("volume", 0), reverse=True)
    return result


@app.get("/api/markets/crypto")
async def api_markets_crypto(request: Request):
    """Crypto market browser — all tracked altcoins with live data."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    now = time.time()
    cached = _markets_crypto_cache.get("data")
    if cached and now - _markets_crypto_cache.get("ts", 0) < 30:
        return {"markets": cached}

    loop = asyncio.get_event_loop()
    markets = await loop.run_in_executor(None, _fetch_crypto_markets_sync)
    _markets_crypto_cache["data"] = markets
    _markets_crypto_cache["ts"] = now
    return {"markets": markets}


_STOCK_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon",
    "NVDA": "NVIDIA", "META": "Meta Platforms", "TSLA": "Tesla", "AMD": "AMD",
    "NFLX": "Netflix", "CRM": "Salesforce", "AVGO": "Broadcom", "ORCL": "Oracle",
    "PLTR": "Palantir", "COIN": "Coinbase", "SQ": "Block", "SHOP": "Shopify",
    "UBER": "Uber", "ABNB": "Airbnb", "SNOW": "Snowflake", "MSTR": "MicroStrategy",
}


def _fetch_stocks_sync() -> list[dict]:
    """Fetch stock prices via free Yahoo Finance API."""
    from config.settings import STOCK_SYMBOLS
    import urllib.request

    tickers = [s.split("/")[0] for s in STOCK_SYMBOLS]
    result = []

    try:
        symbols_str = ",".join(tickers)
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbols_str}&fields=regularMarketPrice,regularMarketChangePercent,regularMarketVolume,marketCap"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        quotes = {q["symbol"]: q for q in data.get("quoteResponse", {}).get("result", [])}
        for sym_pair in STOCK_SYMBOLS:
            sym = sym_pair.split("/")[0]
            q = quotes.get(sym, {})
            result.append({
                "symbol": sym_pair,
                "name": _STOCK_NAMES.get(sym, sym),
                "price": q.get("regularMarketPrice", 0) or 0,
                "change_pct": round(q.get("regularMarketChangePercent", 0) or 0, 2),
                "volume": (q.get("regularMarketVolume", 0) or 0) * (q.get("regularMarketPrice", 0) or 0),
                "market_cap": q.get("marketCap", 0) or 0,
            })
    except Exception:
        # Fallback: return stock list with names but no live data
        for sym_pair in STOCK_SYMBOLS:
            sym = sym_pair.split("/")[0]
            result.append({
                "symbol": sym_pair,
                "name": _STOCK_NAMES.get(sym, sym),
                "price": 0, "change_pct": 0, "volume": 0, "market_cap": 0,
            })

    return result


@app.get("/api/markets/stocks")
async def api_markets_stocks(request: Request):
    """Stock market browser — tracked equities."""
    user = _get_user(request)
    if not user:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    now = time.time()
    cached = _markets_stocks_cache.get("data")
    if cached and now - _markets_stocks_cache.get("ts", 0) < 60:
        return {"markets": cached}

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _fetch_stocks_sync)
    _markets_stocks_cache["data"] = result
    _markets_stocks_cache["ts"] = now
    return {"markets": result}


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
