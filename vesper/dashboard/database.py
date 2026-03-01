"""User database — SQLite with encrypted API key storage."""

import base64
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass

import bcrypt
import pyotp
from cryptography.fernet import Fernet

DATA_DIR = os.environ.get("VESPER_DATA_DIR", "data")
DB_PATH = os.path.join(DATA_DIR, "vesper.db")

# Encryption key for API secrets — generated once and stored
_ENCRYPTION_KEY_FILE = os.path.join(DATA_DIR, ".encryption_key")

_fernet_instance: Fernet | None = None


def _get_fernet() -> Fernet:
    global _fernet_instance
    if _fernet_instance is not None:
        return _fernet_instance
    if os.path.exists(_ENCRYPTION_KEY_FILE):
        with open(_ENCRYPTION_KEY_FILE, "rb") as f:
            key = f.read()
    else:
        os.makedirs(DATA_DIR, exist_ok=True)
        key = Fernet.generate_key()
        with open(_ENCRYPTION_KEY_FILE, "wb") as f:
            f.write(key)
        os.chmod(_ENCRYPTION_KEY_FILE, 0o600)
    _fernet_instance = Fernet(key)
    return _fernet_instance


def _encrypt(value: str) -> str:
    if not value:
        return ""
    return _get_fernet().encrypt(value.encode()).decode()


def _decrypt(value: str) -> str:
    if not value:
        return ""
    return _get_fernet().decrypt(value.encode()).decode()


@dataclass
class User:
    id: int
    email: str
    password_hash: str
    totp_secret: str
    coinbase_api_key: str  # encrypted
    coinbase_api_secret: str  # encrypted
    alpaca_api_key: str  # encrypted
    alpaca_api_secret: str  # encrypted
    kalshi_api_key: str  # encrypted
    kalshi_api_secret: str  # encrypted
    perplexity_api_key: str  # encrypted
    paper_balance: float
    trading_mode: str
    symbols: str
    stop_loss_pct: float
    take_profit_min_pct: float
    take_profit_max_pct: float
    max_position_pct: float
    interval_minutes: int
    bot_active: bool
    created_at: float
    is_admin: bool = False

    def get_api_key(self) -> str:
        return _decrypt(self.coinbase_api_key) if self.coinbase_api_key else ""

    def get_api_secret(self) -> str:
        return _decrypt(self.coinbase_api_secret) if self.coinbase_api_secret else ""

    def get_alpaca_key(self) -> str:
        return _decrypt(self.alpaca_api_key) if self.alpaca_api_key else ""

    def get_alpaca_secret(self) -> str:
        return _decrypt(self.alpaca_api_secret) if self.alpaca_api_secret else ""

    def get_kalshi_key(self) -> str:
        return _decrypt(self.kalshi_api_key) if self.kalshi_api_key else ""

    def get_kalshi_secret(self) -> str:
        return _decrypt(self.kalshi_api_secret) if self.kalshi_api_secret else ""

    def get_perplexity_key(self) -> str:
        return _decrypt(self.perplexity_api_key) if self.perplexity_api_key else ""

    @property
    def has_perplexity(self) -> bool:
        return bool(self.perplexity_api_key)

    @property
    def has_alpaca(self) -> bool:
        return bool(self.alpaca_api_key)

    @property
    def has_coinbase(self) -> bool:
        return bool(self.coinbase_api_key)

    @property
    def has_kalshi(self) -> bool:
        return bool(self.kalshi_api_key)


def _get_conn() -> sqlite3.Connection:
    """Get a SQLite connection with WAL mode and a reasonable busy timeout."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Initialize the database schema."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            totp_secret TEXT NOT NULL,
            coinbase_api_key TEXT DEFAULT '',
            coinbase_api_secret TEXT DEFAULT '',
            paper_balance REAL DEFAULT 500.0,
            trading_mode TEXT DEFAULT 'paper',
            symbols TEXT DEFAULT 'BTC/USDT,ETH/USDT',
            stop_loss_pct REAL DEFAULT 2.0,
            take_profit_min_pct REAL DEFAULT 1.5,
            take_profit_max_pct REAL DEFAULT 5.0,
            max_position_pct REAL DEFAULT 30.0,
            interval_minutes INTEGER DEFAULT 60,
            bot_active INTEGER DEFAULT 0,
            created_at REAL NOT NULL
        )
    """)
    # Migration: add Alpaca columns if missing
    try:
        conn.execute("ALTER TABLE users ADD COLUMN alpaca_api_key TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE users ADD COLUMN alpaca_api_secret TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    # Migration: add Kalshi columns if missing
    try:
        conn.execute("ALTER TABLE users ADD COLUMN kalshi_api_key TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE users ADD COLUMN kalshi_api_secret TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    # Migration: add Perplexity column if missing
    try:
        conn.execute("ALTER TABLE users ADD COLUMN perplexity_api_key TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    # Migration: add is_admin column if missing
    try:
        conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass

    conn.execute("""
        CREATE TABLE IF NOT EXISTS trusted_devices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token_hash TEXT NOT NULL,
            expires_at REAL NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cost_usd REAL DEFAULT 0.0,
            endpoint TEXT DEFAULT '',
            created_at REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def _row_to_user(row: sqlite3.Row) -> User:
    """Convert a sqlite3.Row to a User dataclass."""
    keys = row.keys()
    return User(
        id=row["id"],
        email=row["email"],
        password_hash=row["password_hash"],
        totp_secret=row["totp_secret"],
        coinbase_api_key=row["coinbase_api_key"],
        coinbase_api_secret=row["coinbase_api_secret"],
        alpaca_api_key=row["alpaca_api_key"] if "alpaca_api_key" in keys else "",
        alpaca_api_secret=row["alpaca_api_secret"] if "alpaca_api_secret" in keys else "",
        kalshi_api_key=row["kalshi_api_key"] if "kalshi_api_key" in keys else "",
        kalshi_api_secret=row["kalshi_api_secret"] if "kalshi_api_secret" in keys else "",
        perplexity_api_key=row["perplexity_api_key"] if "perplexity_api_key" in keys else "",
        paper_balance=row["paper_balance"],
        trading_mode=row["trading_mode"],
        symbols=row["symbols"],
        stop_loss_pct=row["stop_loss_pct"],
        take_profit_min_pct=row["take_profit_min_pct"],
        take_profit_max_pct=row["take_profit_max_pct"],
        max_position_pct=row["max_position_pct"],
        interval_minutes=row["interval_minutes"],
        bot_active=bool(row["bot_active"]),
        created_at=row["created_at"],
        is_admin=bool(row["is_admin"]) if "is_admin" in keys else False,
    )


def add_trusted_device(user_id: int, token_hash: str, expires_at: float):
    """Store a trusted device token hash for skipping 2FA."""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO trusted_devices (user_id, token_hash, expires_at, created_at) VALUES (?, ?, ?, ?)",
        (user_id, token_hash, expires_at, time.time()),
    )
    conn.commit()
    conn.close()


def is_device_trusted(user_id: int, token_hash: str) -> bool:
    """Check if a trust token is valid for this user. Prunes expired rows."""
    conn = _get_conn()
    now = time.time()
    conn.execute("DELETE FROM trusted_devices WHERE expires_at < ?", (now,))
    conn.commit()
    row = conn.execute(
        "SELECT id FROM trusted_devices WHERE user_id = ? AND token_hash = ? AND expires_at > ?",
        (user_id, token_hash, now),
    ).fetchone()
    conn.close()
    return row is not None


def remove_trusted_device(token_hash: str):
    """Remove a trusted device token (on logout)."""
    conn = _get_conn()
    conn.execute("DELETE FROM trusted_devices WHERE token_hash = ?", (token_hash,))
    conn.commit()
    conn.close()


def create_user(email: str, password: str, totp_secret: str) -> User | None:
    """Create a new user. Returns None if email already exists."""
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO users (email, password_hash, totp_secret, created_at) VALUES (?, ?, ?, ?)",
            (email.lower().strip(), pw_hash, totp_secret, time.time()),
        )
        conn.commit()
        return get_user_by_email(email)
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def get_user_by_email(email: str) -> User | None:
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()
    conn.close()
    if not row:
        return None
    return _row_to_user(row)


def get_user_by_id(user_id: int) -> User | None:
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return _row_to_user(row)


def verify_password(user: User, password: str) -> bool:
    return bcrypt.checkpw(password.encode(), user.password_hash.encode())


def update_password(user_id: int, new_password_hash: str):
    """Update a user's password hash."""
    conn = _get_conn()
    conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_password_hash, user_id))
    conn.commit()
    conn.close()


def update_api_keys(user_id: int, api_key: str, api_secret: str):
    conn = _get_conn()
    conn.execute(
        "UPDATE users SET coinbase_api_key = ?, coinbase_api_secret = ? WHERE id = ?",
        (_encrypt(api_key), _encrypt(api_secret), user_id),
    )
    conn.commit()
    conn.close()


def update_alpaca_keys(user_id: int, api_key: str, api_secret: str):
    conn = _get_conn()
    conn.execute(
        "UPDATE users SET alpaca_api_key = ?, alpaca_api_secret = ? WHERE id = ?",
        (_encrypt(api_key), _encrypt(api_secret), user_id),
    )
    conn.commit()
    conn.close()


def update_kalshi_keys(user_id: int, api_key: str, api_secret: str):
    conn = _get_conn()
    conn.execute(
        "UPDATE users SET kalshi_api_key = ?, kalshi_api_secret = ? WHERE id = ?",
        (_encrypt(api_key), _encrypt(api_secret), user_id),
    )
    conn.commit()
    conn.close()


def update_perplexity_key(user_id: int, api_key: str):
    conn = _get_conn()
    conn.execute(
        "UPDATE users SET perplexity_api_key = ? WHERE id = ?",
        (_encrypt(api_key), user_id),
    )
    conn.commit()
    conn.close()


def update_trading_config(
    user_id: int,
    paper_balance: float,
    trading_mode: str,
    symbols: str,
    stop_loss_pct: float,
    take_profit_min_pct: float,
    take_profit_max_pct: float,
    max_position_pct: float,
    interval_minutes: int,
):
    conn = _get_conn()
    conn.execute(
        """UPDATE users SET
            paper_balance = ?, trading_mode = ?, symbols = ?,
            stop_loss_pct = ?, take_profit_min_pct = ?,
            take_profit_max_pct = ?, max_position_pct = ?,
            interval_minutes = ?
        WHERE id = ?""",
        (paper_balance, trading_mode, symbols, stop_loss_pct,
         take_profit_min_pct, take_profit_max_pct, max_position_pct,
         interval_minutes, user_id),
    )
    conn.commit()
    conn.close()


def set_bot_active(user_id: int, active: bool):
    conn = _get_conn()
    conn.execute("UPDATE users SET bot_active = ? WHERE id = ?", (int(active), user_id))
    conn.commit()
    conn.close()


def create_oauth_user(email: str) -> User | None:
    """Create a user from OAuth sign-in (no password/TOTP required)."""
    placeholder_hash = bcrypt.hashpw(secrets.token_hex(32).encode(), bcrypt.gensalt()).decode()
    placeholder_totp = pyotp.random_base32()
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO users (email, password_hash, totp_secret, created_at) VALUES (?, ?, ?, ?)",
            (email.lower().strip(), placeholder_hash, placeholder_totp, time.time()),
        )
        conn.commit()
        return get_user_by_email(email)
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def update_trading_mode(user_id: int, mode: str):
    """Update just the trading mode (paper/live)."""
    conn = _get_conn()
    conn.execute("UPDATE users SET trading_mode = ? WHERE id = ?", (mode, user_id))
    conn.commit()
    conn.close()


def get_active_users() -> list[User]:
    """Get all users with active bots."""
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM users WHERE bot_active = 1").fetchall()
    conn.close()
    return [_row_to_user(row) for row in rows]


def get_all_users() -> list[User]:
    """Get all registered users (admin only)."""
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    conn.close()
    return [_row_to_user(row) for row in rows]


def set_admin(user_id: int, is_admin: bool):
    """Promote/demote a user to admin."""
    conn = _get_conn()
    conn.execute("UPDATE users SET is_admin = ? WHERE id = ?", (int(is_admin), user_id))
    conn.commit()
    conn.close()


# ── API usage tracking ──

def log_api_usage(
    provider: str, model: str,
    input_tokens: int, output_tokens: int,
    cost_usd: float, endpoint: str = "",
    user_id: int | None = None,
):
    """Record an API call for cost tracking."""
    import logging
    _log = logging.getLogger(__name__)
    _log.info(
        f"[api_usage] {provider}/{model} — {input_tokens}in/{output_tokens}out "
        f"${cost_usd:.6f} ({endpoint})"
    )
    conn = _get_conn()
    conn.execute(
        """INSERT INTO api_usage
           (user_id, provider, model, input_tokens, output_tokens, cost_usd, endpoint, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, provider, model, input_tokens, output_tokens, cost_usd, endpoint, time.time()),
    )
    conn.commit()
    conn.close()


def get_api_usage_summary(days: int = 30) -> dict:
    """Aggregate API costs for the admin dashboard."""
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    cutoff = time.time() - (days * 86400)

    # Total costs by provider
    rows = conn.execute(
        """SELECT provider, model,
                  SUM(input_tokens) as total_input,
                  SUM(output_tokens) as total_output,
                  SUM(cost_usd) as total_cost,
                  COUNT(*) as call_count
           FROM api_usage WHERE created_at > ?
           GROUP BY provider, model
           ORDER BY total_cost DESC""",
        (cutoff,),
    ).fetchall()

    by_model = [dict(r) for r in rows]
    total_cost = sum(r["total_cost"] for r in by_model)
    total_calls = sum(r["call_count"] for r in by_model)

    # Daily breakdown (last N days)
    daily = conn.execute(
        """SELECT date(created_at, 'unixepoch') as day,
                  provider,
                  SUM(cost_usd) as cost,
                  COUNT(*) as calls
           FROM api_usage WHERE created_at > ?
           GROUP BY day, provider
           ORDER BY day DESC""",
        (cutoff,),
    ).fetchall()

    # Recent calls
    recent = conn.execute(
        """SELECT provider, model, input_tokens, output_tokens,
                  cost_usd, endpoint, created_at
           FROM api_usage ORDER BY created_at DESC LIMIT 50""",
    ).fetchall()

    conn.close()
    return {
        "total_cost": round(total_cost, 4),
        "total_calls": total_calls,
        "by_model": by_model,
        "daily": [dict(r) for r in daily],
        "recent": [dict(r) for r in recent],
    }
