"""Exchange connectors — Coinbase (ccxt) + Alpaca (alpaca-py)."""

import ccxt
from config.settings import ExchangeConfig


# ── Coinbase (crypto) via ccxt ──────────────────────────────

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


# ── Alpaca (stocks) ─────────────────────────────────────────

class AlpacaExchange:
    """Wrapper around alpaca-py that provides a consistent interface."""

    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient

        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.trading = TradingClient(api_key, api_secret, paper=paper)
        self.data = StockHistoricalDataClient(api_key, api_secret)
        self.id = "alpaca"

    def fetch_ticker(self, symbol: str) -> dict:
        """Fetch latest quote for a stock symbol (e.g. 'AAPL')."""
        from alpaca.data.requests import StockLatestQuoteRequest
        # Strip /USD suffix if present (e.g. "AAPL/USD" -> "AAPL")
        ticker = symbol.split("/")[0]
        req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
        quotes = self.data.get_stock_latest_quote(req)
        quote = quotes[ticker]
        mid = (quote.ask_price + quote.bid_price) / 2
        return {
            "symbol": symbol,
            "last": mid,
            "bid": quote.bid_price,
            "ask": quote.ask_price,
            "close": mid,
        }

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1Hour", limit: int = 100):
        """Fetch OHLCV bars for a stock. Returns list of [timestamp, O, H, L, C, V]."""
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from datetime import datetime, timedelta

        ticker = symbol.split("/")[0]
        tf_map = {
            "1m": TimeFrame.Minute,
            "5m": TimeFrame(5, TimeFrameUnit.Minute),
            "15m": TimeFrame(15, TimeFrameUnit.Minute),
            "1h": TimeFrame.Hour,
            "4h": TimeFrame(4, TimeFrameUnit.Hour),
            "1d": TimeFrame.Day,
        }
        tf = tf_map.get(timeframe, TimeFrame.Hour)
        start = datetime.now() - timedelta(days=max(limit // 6, 30))

        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=tf,
            start=start,
            limit=limit,
        )
        bars = self.data.get_stock_bars(req)
        result = []
        for bar in bars[ticker]:
            result.append([
                int(bar.timestamp.timestamp() * 1000),
                bar.open, bar.high, bar.low, bar.close, bar.volume,
            ])
        return result

    def get_account(self) -> dict:
        """Get account info (buying power, equity, etc.)."""
        account = self.trading.get_account()
        return {
            "buying_power": float(account.buying_power),
            "equity": float(account.equity),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
        }

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        positions = self.trading.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            }
            for p in positions
        ]


def alpaca_market_buy(client: AlpacaExchange, symbol: str, notional: float) -> dict:
    """Place a market buy order for a dollar amount of stock."""
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    ticker = symbol.split("/")[0]
    req = MarketOrderRequest(
        symbol=ticker,
        notional=round(notional, 2),
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    order = client.trading.submit_order(req)
    return {
        "id": str(order.id),
        "symbol": ticker,
        "filled": float(order.filled_qty or 0),
        "average": float(order.filled_avg_price or 0),
        "status": str(order.status),
    }


def alpaca_market_sell(client: AlpacaExchange, symbol: str, qty: float) -> dict:
    """Place a market sell order for a quantity of stock."""
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    ticker = symbol.split("/")[0]
    req = MarketOrderRequest(
        symbol=ticker,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    order = client.trading.submit_order(req)
    return {
        "id": str(order.id),
        "symbol": ticker,
        "filled": float(order.filled_qty or 0),
        "average": float(order.filled_avg_price or 0),
        "status": str(order.status),
    }
