"""Exchange connector — wraps Coinbase Advanced via ccxt."""

import ccxt
from config.settings import ExchangeConfig


def create_exchange(cfg: ExchangeConfig) -> ccxt.coinbase:
    """Create and return a configured Coinbase Advanced exchange instance."""
    exchange = ccxt.coinbase(
        {
            "apiKey": cfg.api_key,
            "secret": cfg.api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
    )
    if cfg.sandbox:
        exchange.set_sandbox_mode(True)
    return exchange


def fetch_ticker(exchange: ccxt.Exchange, symbol: str) -> dict:
    """Fetch current ticker for a symbol."""
    return exchange.fetch_ticker(symbol)


def fetch_order_book(exchange: ccxt.Exchange, symbol: str, limit: int = 20) -> dict:
    """Fetch order book for a symbol."""
    return exchange.fetch_order_book(symbol, limit=limit)


def place_market_buy(exchange: ccxt.Exchange, symbol: str, amount: float) -> dict:
    """Place a market buy order."""
    return exchange.create_market_buy_order(symbol, amount)


def place_market_sell(exchange: ccxt.Exchange, symbol: str, amount: float) -> dict:
    """Place a market sell order."""
    return exchange.create_market_sell_order(symbol, amount)


def fetch_balance(exchange: ccxt.Exchange) -> dict:
    """Fetch account balance."""
    return exchange.fetch_balance()
