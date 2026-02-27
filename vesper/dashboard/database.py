"""User database — SQLite with encrypted API key storage."""

import base64
import os
import sqlite3
import time
from dataclasses import dataclass

import bcrypt
from cryptography.fernet import Fernet

DATA_DIR = os.environ.get("VESPER_DATA_DIR", "data")
DB_PATH = os.path.join(DATA_DIR, "vesper.db")

# Encryption key for API secrets — generated once and stored
_ENCRYPTION_KEY_FILE = os.path.join(DATA_DIR, ".encryption_key")


def _get_fernet() -> Fernet:
    if os.path.exists(_ENCRYPTION_KEY_FILE):
        with open(_ENCRYPTION_KEY_FILE, "rb") as f:
            key = f.read()
    else:
        os.makedirs(DATA_DIR, exist_ok=True)
        key = Fernet.generate_key()
        with open(_ENCRYPTION_KEY_FILE, "wb") as f:
            f.write(key)
        os.chmod(_ENCRYPTION_KEY_FILE, 0o600)
    return Fernet(key)


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

    def get_api_key(self) -> str:
        return _decrypt(self.coinbase_api_key) if self.coinbase_api_key else ""

    def get_api_secret(self) -> str:
        return _decrypt(self.coinbase_api_secret) if self.coinbase_api_secret else ""


def init_db():
    """Initialize the database schema."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
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
    conn.commit()
    conn.close()


def create_user(email: str, password: str, totp_secret: str) -> User | None:
    """Create a new user. Returns None if email already exists."""
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()
    conn.close()
    if not row:
        return None
    return User(
        id=row["id"],
        email=row["email"],
        password_hash=row["password_hash"],
        totp_secret=row["totp_secret"],
        coinbase_api_key=row["coinbase_api_key"],
        coinbase_api_secret=row["coinbase_api_secret"],
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
    )


def get_user_by_id(user_id: int) -> User | None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return User(
        id=row["id"],
        email=row["email"],
        password_hash=row["password_hash"],
        totp_secret=row["totp_secret"],
        coinbase_api_key=row["coinbase_api_key"],
        coinbase_api_secret=row["coinbase_api_secret"],
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
    )


def verify_password(user: User, password: str) -> bool:
    return bcrypt.checkpw(password.encode(), user.password_hash.encode())


def update_api_keys(user_id: int, api_key: str, api_secret: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE users SET coinbase_api_key = ?, coinbase_api_secret = ? WHERE id = ?",
        (_encrypt(api_key), _encrypt(api_secret), user_id),
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
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE users SET bot_active = ? WHERE id = ?", (int(active), user_id))
    conn.commit()
    conn.close()


def get_active_users() -> list[User]:
    """Get all users with active bots."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM users WHERE bot_active = 1").fetchall()
    conn.close()
    users = []
    for row in rows:
        users.append(User(
            id=row["id"],
            email=row["email"],
            password_hash=row["password_hash"],
            totp_secret=row["totp_secret"],
            coinbase_api_key=row["coinbase_api_key"],
            coinbase_api_secret=row["coinbase_api_secret"],
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
        ))
    return users
