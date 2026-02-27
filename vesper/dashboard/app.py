"""Vesper Dashboard — Web UI with password + TOTP 2FA authentication."""

import base64
import hmac
import io
import os
import json
import secrets
import time
from datetime import datetime

import pyotp
import qrcode
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Vesper Dashboard", docs_url=None, redoc_url=None)

templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)

DATA_DIR = os.environ.get("VESPER_DATA_DIR", "data")
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "changeme123")

# 2FA state
_TOTP_SECRET_FILE = os.path.join(DATA_DIR, ".totp_secret")
_valid_sessions: set[str] = set()

# Rate limiting: track failed attempts per IP
_failed_attempts: dict[str, list[float]] = {}
MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 300  # 5 minutes


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _is_rate_limited(ip: str) -> bool:
    now = time.time()
    attempts = _failed_attempts.get(ip, [])
    # Keep only recent attempts
    attempts = [t for t in attempts if now - t < LOCKOUT_SECONDS]
    _failed_attempts[ip] = attempts
    return len(attempts) >= MAX_ATTEMPTS


def _record_failed_attempt(ip: str):
    now = time.time()
    if ip not in _failed_attempts:
        _failed_attempts[ip] = []
    _failed_attempts[ip].append(now)


def _clear_failed_attempts(ip: str):
    _failed_attempts.pop(ip, None)


def _get_totp_secret() -> str | None:
    if os.path.exists(_TOTP_SECRET_FILE):
        with open(_TOTP_SECRET_FILE) as f:
            return f.read().strip()
    return None


def _save_totp_secret(secret: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(_TOTP_SECRET_FILE, "w") as f:
        f.write(secret)
    os.chmod(_TOTP_SECRET_FILE, 0o600)


def _is_2fa_enabled() -> bool:
    return _get_totp_secret() is not None


def _check_session(request: Request) -> bool:
    token = request.cookies.get("vesper_session")
    return token is not None and token in _valid_sessions


def _create_session() -> tuple[str, RedirectResponse]:
    token = secrets.token_hex(32)
    _valid_sessions.add(token)
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        key="vesper_session",
        value=token,
        httponly=True,
        max_age=86400 * 7,
        samesite="strict",
    )
    return token, response


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


# --- Auth routes ---


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = ""):
    ip = _get_client_ip(request)
    locked = _is_rate_limited(ip)
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": error,
        "locked": locked,
        "has_2fa": _is_2fa_enabled(),
    })


@app.post("/login")
async def login(
    request: Request,
    password: str = Form(...),
    totp_code: str = Form(""),
):
    ip = _get_client_ip(request)

    if _is_rate_limited(ip):
        return RedirectResponse(url="/login?error=locked", status_code=303)

    if not hmac.compare_digest(password, DASHBOARD_PASSWORD):
        _record_failed_attempt(ip)
        return RedirectResponse(url="/login?error=wrong_password", status_code=303)

    # Check 2FA if enabled
    if _is_2fa_enabled():
        secret = _get_totp_secret()
        totp = pyotp.TOTP(secret)
        if not totp.verify(totp_code, valid_window=1):
            _record_failed_attempt(ip)
            return RedirectResponse(url="/login?error=wrong_2fa", status_code=303)

    _clear_failed_attempts(ip)
    _, response = _create_session()
    return response


@app.get("/logout")
async def logout(request: Request):
    token = request.cookies.get("vesper_session")
    if token:
        _valid_sessions.discard(token)
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("vesper_session")
    return response


# --- 2FA setup routes ---


@app.get("/setup-2fa", response_class=HTMLResponse)
async def setup_2fa_page(request: Request):
    if not _check_session(request):
        return RedirectResponse(url="/login", status_code=303)

    if _is_2fa_enabled():
        return RedirectResponse(url="/", status_code=303)

    # Generate new secret
    secret = pyotp.random_base32()
    totp = pyotp.TOTP(secret)
    provisioning_uri = totp.provisioning_uri(
        name="admin",
        issuer_name="Vesper Trading Bot",
    )

    # Generate QR code as base64
    img = qrcode.make(provisioning_uri)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    qr_b64 = base64.b64encode(buffer.getvalue()).decode()

    return templates.TemplateResponse("setup_2fa.html", {
        "request": request,
        "secret": secret,
        "qr_b64": qr_b64,
    })


@app.post("/setup-2fa")
async def setup_2fa_confirm(
    request: Request,
    secret: str = Form(...),
    totp_code: str = Form(...),
):
    if not _check_session(request):
        return RedirectResponse(url="/login", status_code=303)

    # Verify the code before saving
    totp = pyotp.TOTP(secret)
    if not totp.verify(totp_code, valid_window=1):
        return RedirectResponse(url="/setup-2fa?error=invalid", status_code=303)

    _save_totp_secret(secret)
    return RedirectResponse(url="/?msg=2fa_enabled", status_code=303)


@app.get("/disable-2fa", response_class=HTMLResponse)
async def disable_2fa_page(request: Request):
    if not _check_session(request):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("disable_2fa.html", {"request": request})


@app.post("/disable-2fa")
async def disable_2fa_confirm(
    request: Request,
    password: str = Form(...),
    totp_code: str = Form(...),
):
    if not _check_session(request):
        return RedirectResponse(url="/login", status_code=303)

    if not hmac.compare_digest(password, DASHBOARD_PASSWORD):
        return RedirectResponse(url="/disable-2fa?error=wrong", status_code=303)

    secret = _get_totp_secret()
    if secret:
        totp = pyotp.TOTP(secret)
        if not totp.verify(totp_code, valid_window=1):
            return RedirectResponse(url="/disable-2fa?error=wrong", status_code=303)
        os.remove(_TOTP_SECRET_FILE)

    return RedirectResponse(url="/?msg=2fa_disabled", status_code=303)


# --- Dashboard routes ---


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if not _check_session(request):
        return RedirectResponse(url="/login", status_code=303)

    portfolio = _load_portfolio()

    cash = portfolio.get("cash", 0)
    initial = portfolio.get("initial_balance", 0)
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
        exit_time = t.get("exit_time", 0)
        equity_curve.append({
            "time": exit_time,
            "value": round(running, 2),
        })

    formatted_trades = []
    for t in reversed(trades[-50:]):
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
        "has_2fa": _is_2fa_enabled(),
    })


@app.get("/api/portfolio")
async def api_portfolio(request: Request):
    if not _check_session(request):
        return {"error": "unauthorized"}, 401
    return _load_portfolio()


@app.get("/api/health")
async def health():
    return {"status": "running", "time": datetime.now().isoformat()}
