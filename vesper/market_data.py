"""Market data fetcher and technical indicator calculator."""

import ccxt
import pandas as pd
import ta


def fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
) -> pd.DataFrame:
    """Fetch OHLCV candles and return as a DataFrame."""
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to an OHLCV DataFrame."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Moving averages
    df["ema_12"] = ta.trend.ema_indicator(close, window=12)
    df["ema_26"] = ta.trend.ema_indicator(close, window=26)
    df["sma_50"] = ta.trend.sma_indicator(close, window=50)

    # RSI
    df["rsi"] = ta.momentum.rsi(close, window=14)

    # MACD
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()

    # ATR (for dynamic stop-loss)
    df["atr"] = ta.volatility.average_true_range(high, low, close, window=14)

    # Volume SMA
    df["volume_sma"] = df["volume"].rolling(window=20).mean()

    return df


def get_market_snapshot(exchange: ccxt.Exchange, symbol: str) -> dict:
    """Get a complete market snapshot with indicators."""
    df = fetch_ohlcv(exchange, symbol, timeframe="1h", limit=100)
    df = add_indicators(df)
    latest = df.iloc[-1]

    return {
        "symbol": symbol,
        "price": latest["close"],
        "ema_12": latest["ema_12"],
        "ema_26": latest["ema_26"],
        "rsi": latest["rsi"],
        "macd": latest["macd"],
        "macd_signal": latest["macd_signal"],
        "macd_hist": latest["macd_hist"],
        "bb_upper": latest["bb_upper"],
        "bb_lower": latest["bb_lower"],
        "atr": latest["atr"],
        "volume": latest["volume"],
        "volume_sma": latest["volume_sma"],
        "df": df,
    }
