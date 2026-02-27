"""Market data fetcher and technical indicator calculator."""

import time

import ccxt
import httpx
import pandas as pd
import ta

from vesper.whale_tracker import fetch_whale_activity
from vesper.sentiment import fetch_composite_sentiment


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
    volume = df["volume"]

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
    df["volume_sma"] = volume.rolling(window=20).mean()

    # --- New indicators ---

    # VWAP (Volume Weighted Average Price)
    tp = (high + low + close) / 3
    df["vwap"] = (tp * volume).cumsum() / volume.cumsum()

    # OBV (On-Balance Volume) — confirms trend with volume flow
    df["obv"] = ta.volume.on_balance_volume(close, volume)

    # ADX (Average Directional Index) — trend strength: >25 strong, <20 weak
    df["adx"] = ta.trend.adx(high, low, close, window=14)

    # Stochastic RSI — more sensitive overbought/oversold
    df["stoch_rsi_k"] = ta.momentum.stochrsi_k(close, window=14, smooth1=3, smooth2=3)
    df["stoch_rsi_d"] = ta.momentum.stochrsi_d(close, window=14, smooth1=3, smooth2=3)

    return df


def _extract_snapshot(df: pd.DataFrame, symbol: str) -> dict:
    """Extract latest indicator values from a DataFrame into a snapshot dict."""
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
        "vwap": latest.get("vwap", latest["close"]),
        "obv": latest.get("obv", 0),
        "adx": latest.get("adx", 0),
        "stoch_rsi_k": latest.get("stoch_rsi_k", 50),
        "stoch_rsi_d": latest.get("stoch_rsi_d", 50),
        "df": df,
    }


def get_market_snapshot(
    exchange: ccxt.Exchange, symbol: str, timeframe: str = "1h"
) -> dict:
    """Get a complete market snapshot with indicators for a single timeframe."""
    df = fetch_ohlcv(exchange, symbol, timeframe=timeframe, limit=100)
    df = add_indicators(df)
    return _extract_snapshot(df, symbol)


def get_multi_tf_snapshot(exchange: ccxt.Exchange, symbol: str) -> dict:
    """Get market snapshot with 1h + 4h data for multi-timeframe alignment.

    Returns the 1h snapshot enriched with:
    - tf_4h: dict of 4h indicator values
    - tf_alignment: 0.0 (bearish both) to 1.0 (bullish both)
    - adx_4h: 4h ADX for trend strength confirmation
    """
    # Primary: 1h candles
    snapshot = get_market_snapshot(exchange, symbol, timeframe="1h")

    # Secondary: 4h candles
    try:
        df_4h = fetch_ohlcv(exchange, symbol, timeframe="4h", limit=100)
        df_4h = add_indicators(df_4h)
        latest_4h = df_4h.iloc[-1]

        ema_bullish_1h = snapshot["ema_12"] > snapshot["ema_26"]
        ema_bullish_4h = latest_4h["ema_12"] > latest_4h["ema_26"]

        # Alignment score
        if ema_bullish_1h and ema_bullish_4h:
            alignment = 1.0
        elif not ema_bullish_1h and not ema_bullish_4h:
            alignment = 0.0
        else:
            alignment = 0.5  # Mixed signals

        snapshot["tf_4h"] = {
            "ema_12": latest_4h["ema_12"],
            "ema_26": latest_4h["ema_26"],
            "rsi": latest_4h["rsi"],
            "macd_hist": latest_4h["macd_hist"],
            "adx": latest_4h.get("adx", 0),
        }
        snapshot["tf_alignment"] = alignment
        snapshot["adx_4h"] = latest_4h.get("adx", 0)
    except Exception:
        snapshot["tf_4h"] = {}
        snapshot["tf_alignment"] = 0.5
        snapshot["adx_4h"] = 0

    return snapshot


def enrich_with_intelligence(exchange: ccxt.Exchange, symbol: str, snapshot: dict) -> dict:
    """Enrich a snapshot with whale tracking + composite sentiment.

    Adds to snapshot:
        whale_score: -1.0 to +1.0 (whale buying/selling pressure)
        whale_details: list of whale activity descriptions
        sentiment_score: -1.0 to +1.0 (composite crowd sentiment)
        sentiment_details: list of sentiment source descriptions
    """
    # Whale tracking
    try:
        whale = fetch_whale_activity(exchange, symbol)
        snapshot["whale_score"] = whale["whale_score"]
        snapshot["whale_details"] = whale["details"]
        snapshot["large_buys"] = whale["large_buys"]
        snapshot["large_sells"] = whale["large_sells"]
        snapshot["volume_anomaly"] = whale["volume_anomaly"]
    except Exception:
        snapshot["whale_score"] = 0.0
        snapshot["whale_details"] = []

    # Composite sentiment (uses fear_greed if already in snapshot)
    try:
        fg = snapshot.get("fear_greed", 50)
        sentiment = fetch_composite_sentiment(symbol, fear_greed=fg)
        snapshot["sentiment_score"] = sentiment["composite_score"]
        snapshot["sentiment_details"] = sentiment["details"]
        snapshot["reddit_score"] = sentiment.get("reddit_score", 0.0)
        snapshot["momentum_score"] = sentiment.get("momentum_score", 0.0)
    except Exception:
        snapshot["sentiment_score"] = 0.0
        snapshot["sentiment_details"] = []

    return snapshot


def get_order_book_pressure(exchange: ccxt.Exchange, symbol: str) -> dict:
    """Analyze order book bid/ask depth for buy/sell pressure.

    Returns:
        buy_pressure: 0.0-1.0 (>0.6 bullish, <0.4 bearish)
        spread_pct: bid-ask spread as percentage
    """
    try:
        book = exchange.fetch_order_book(symbol, limit=20)
        bids = book.get("bids", [])
        asks = book.get("asks", [])

        if not bids or not asks:
            return {"buy_pressure": 0.5, "spread_pct": 0.0}

        bid_volume = sum(b[1] for b in bids)
        ask_volume = sum(a[1] for a in asks)
        total = bid_volume + ask_volume

        buy_pressure = bid_volume / total if total > 0 else 0.5
        spread_pct = ((asks[0][0] - bids[0][0]) / bids[0][0]) * 100 if bids[0][0] > 0 else 0

        return {
            "buy_pressure": round(buy_pressure, 3),
            "spread_pct": round(spread_pct, 4),
        }
    except Exception:
        return {"buy_pressure": 0.5, "spread_pct": 0.0}


# --- Fear & Greed Index (cached) ---

_fear_greed_cache: dict = {"value": 50, "time": 0}
_FEAR_GREED_TTL = 1800  # 30 minutes


def fetch_fear_greed() -> int:
    """Fetch crypto Fear & Greed Index (0-100). Cached 30 min.

    0-24: Extreme Fear (contrarian buy signal)
    25-44: Fear
    45-54: Neutral
    55-74: Greed
    75-100: Extreme Greed (cautious / contrarian sell)
    """
    global _fear_greed_cache
    now = time.time()

    if (now - _fear_greed_cache["time"]) < _FEAR_GREED_TTL:
        return _fear_greed_cache["value"]

    try:
        resp = httpx.get("https://api.alternative.me/fng/", timeout=5)
        data = resp.json()
        value = int(data["data"][0]["value"])
        _fear_greed_cache = {"value": value, "time": now}
        return value
    except Exception:
        return _fear_greed_cache["value"]
