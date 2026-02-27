"""Vesper Dashboard — SaaS with OAuth, 2FA, HTTPS."""

import base64
import hmac
import io
import os
import json
import secrets
import time
from datetime import datetime

import bcrypt
import httpx
import pyotp
import qrcode
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from vesper.dashboard.database import (
    init_db, create_user, get_user_by_email, get_user_by_id,
    verify_password, update_api_keys, update_trading_config,
    set_bot_active, update_trading_mode, create_oauth_user, User,
)

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
                password: str = Form(...), totp_code: str = Form("")):
    ip = _get_ip(request)
    if _is_locked(ip):
        return RedirectResponse(url="/login?error=locked", status_code=303)
    user = get_user_by_email(email)
    if not user or not verify_password(user, password):
        _record_fail(ip)
        return RedirectResponse(url="/login?error=invalid", status_code=303)
    if not pyotp.TOTP(user.totp_secret).verify(totp_code, valid_window=1):
        _record_fail(ip)
        return RedirectResponse(url="/login?error=invalid_2fa", status_code=303)
    _clear_fails(ip)
    return _create_session(user.id)


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
    resp = RedirectResponse(url="/", status_code=303)
    resp.delete_cookie("vesper_session")
    return resp


# --- Dashboard ---

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    portfolio = _load_portfolio(user.id)
    cash = portfolio.get("cash", user.paper_balance)
    initial = portfolio.get("initial_balance", user.paper_balance)
    positions = portfolio.get("positions", {})
    trades = portfolio.get("trade_history", [])
    total_pnl = sum(t.get("pnl_usd", 0) for t in trades)
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
        fmt_pos.append({
            "id": pid, "symbol": p.get("symbol", ""), "side": p.get("side", ""),
            "entry_price": p.get("entry_price", 0), "amount": p.get("amount", 0),
            "cost_usd": p.get("cost_usd", 0),
            "entry_time": datetime.fromtimestamp(p.get("entry_time", 0)).strftime("%m/%d %H:%M"),
            "stop_loss": p.get("limits", {}).get("stop_loss_price", 0),
            "tp_min": p.get("limits", {}).get("take_profit_min_price", 0),
            "tp_max": p.get("limits", {}).get("take_profit_max_price", 0),
        })

    return templates.TemplateResponse("dashboard.html", {
        "request": request, "user": user, "cash": cash,
        "initial_balance": initial, "total_pnl": total_pnl,
        "total_pnl_pct": (total_pnl / initial * 100) if initial > 0 else 0,
        "win_rate": win_rate, "total_trades": len(trades),
        "wins": len(wins), "losses": len(losses),
        "positions": fmt_pos, "trades": fmt_trades,
        "equity_curve": json.dumps(equity),
        "has_api_keys": bool(user.coinbase_api_key),
    })


# --- Settings ---

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, msg: str = ""):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("settings.html", {
        "request": request, "user": user,
        "has_api_keys": bool(user.coinbase_api_key), "msg": msg,
    })

@app.post("/settings/api-keys")
async def save_keys(request: Request, api_key: str = Form(...), api_secret: str = Form(...)):
    user = _get_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    update_api_keys(user.id, api_key, api_secret)
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


# --- API ---

@app.get("/api/health")
async def health():
    return {"status": "running", "time": datetime.now().isoformat()}
