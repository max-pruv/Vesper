"""Composite sentiment analysis — aggregates Reddit, CoinGecko momentum, Fear&Greed.

Returns a composite_score from -1.0 (very bearish crowd) to +1.0 (very bullish).
The AI ensemble uses this to confirm or override technical signals.
"""

import logging
import time

import httpx

logger = logging.getLogger(__name__)

# Per-symbol cache
_sentiment_cache: dict[str, dict] = {}
_SENTIMENT_CACHE_TTL = 900  # 15 minutes

BULLISH_KEYWORDS = [
    "bull", "bullish", "moon", "pump", "buy", "long", "breakout",
    "ath", "all time high", "rocket", "surge", "rally", "hodl",
    "accumulate", "support", "bounce", "recovery", "green",
    "undervalued", "dip", "buying opportunity",
]
BEARISH_KEYWORDS = [
    "bear", "bearish", "crash", "dump", "sell", "short", "breakdown",
    "fear", "panic", "red", "capitulation", "collapse", "plunge",
    "resistance", "reject", "top", "overvalued", "bubble",
    "rug", "scam", "dead",
]


def _analyze_text_sentiment(texts: list[str]) -> float:
    """Keyword-based sentiment from text. Returns -1.0 to +1.0."""
    if not texts:
        return 0.0

    bull_count = 0
    bear_count = 0

    for text in texts:
        lower = text.lower()
        for kw in BULLISH_KEYWORDS:
            if kw in lower:
                bull_count += 1
        for kw in BEARISH_KEYWORDS:
            if kw in lower:
                bear_count += 1

    total = bull_count + bear_count
    if total == 0:
        return 0.0

    return (bull_count - bear_count) / total


def _fetch_reddit_sentiment(symbol: str) -> tuple[float, list[str]]:
    """Fetch sentiment from Reddit crypto subreddits. Returns (score, details)."""
    details = []
    texts = []

    # Map symbols to relevant subreddits
    subreddits = ["CryptoCurrency"]
    base = symbol.split("/")[0].upper()
    sub_map = {
        "BTC": "Bitcoin", "ETH": "ethereum", "SOL": "solana",
        "ADA": "cardano", "DOT": "Polkadot", "DOGE": "dogecoin",
        "AVAX": "Avax", "LINK": "Chainlink", "XRP": "XRP",
    }
    if base in sub_map:
        subreddits.append(sub_map[base])

    for sub in subreddits:
        try:
            resp = httpx.get(
                f"https://www.reddit.com/r/{sub}/hot.json?limit=25",
                headers={"User-Agent": "Vesper/1.0 trading bot"},
                timeout=8,
                follow_redirects=True,
            )
            if resp.status_code == 200:
                data = resp.json()
                posts = data.get("data", {}).get("children", [])
                for post in posts:
                    title = post.get("data", {}).get("title", "")
                    if title:
                        texts.append(title)
        except Exception as e:
            logger.debug(f"Reddit fetch failed for r/{sub}: {e}")
            continue

    if texts:
        score = _analyze_text_sentiment(texts)
        label = "bullish" if score > 0.1 else "bearish" if score < -0.1 else "neutral"
        details.append(f"Reddit ({len(texts)} posts): {label} ({score:+.2f})")
        return score, details

    return 0.0, []


_COINGECKO_MAP = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
    "XRP": "ripple", "ADA": "cardano", "DOGE": "dogecoin",
    "DOT": "polkadot", "AVAX": "avalanche-2", "LINK": "chainlink",
    "MATIC": "matic-network", "UNI": "uniswap", "ATOM": "cosmos",
    "LTC": "litecoin", "BCH": "bitcoin-cash", "NEAR": "near",
    "FIL": "filecoin", "APT": "aptos", "ARB": "arbitrum",
    "OP": "optimism", "SHIB": "shiba-inu",
}


def _fetch_coingecko_momentum(symbol: str) -> tuple[float, list[str]]:
    """Get price momentum from CoinGecko as a sentiment proxy. Returns (score, details)."""
    base = symbol.split("/")[0].upper()
    coin_id = _COINGECKO_MAP.get(base)
    if not coin_id:
        return 0.0, []

    try:
        resp = httpx.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}",
            params={
                "localization": "false",
                "tickers": "false",
                "community_data": "false",
                "developer_data": "false",
            },
            timeout=8,
        )
        if resp.status_code != 200:
            return 0.0, []

        market = resp.json().get("market_data", {})
        pct_24h = market.get("price_change_percentage_24h", 0) or 0
        pct_7d = market.get("price_change_percentage_7d", 0) or 0

        # Normalize: ±10% daily = ±1.0, ±20% weekly = ±1.0
        momentum = (pct_24h / 10) * 0.6 + (pct_7d / 20) * 0.4
        momentum = max(-1.0, min(1.0, momentum))

        details = [f"Price momentum: {pct_24h:+.1f}% (24h), {pct_7d:+.1f}% (7d)"]
        return momentum, details
    except Exception as e:
        logger.debug(f"CoinGecko momentum fetch failed for {symbol}: {e}")
        return 0.0, []


def fetch_composite_sentiment(symbol: str, fear_greed: int = 50) -> dict:
    """Aggregate sentiment from multiple sources.

    Returns:
        composite_score: -1.0 (very bearish) to +1.0 (very bullish)
        reddit_score: -1.0 to +1.0
        momentum_score: -1.0 to +1.0
        fear_greed_normalized: -1.0 to +1.0
        details: list of human-readable detail strings
    """
    now = time.time()
    cached = _sentiment_cache.get(symbol)
    if cached and (now - cached["time"]) < _SENTIMENT_CACHE_TTL:
        return cached["data"]

    result = {
        "composite_score": 0.0,
        "reddit_score": 0.0,
        "momentum_score": 0.0,
        "fear_greed_normalized": 0.0,
        "details": [],
    }

    signals = []  # (name, score, weight)

    # Source 1: Fear & Greed (normalize 0-100 → -1 to +1)
    fg_normalized = (fear_greed - 50) / 50
    result["fear_greed_normalized"] = round(fg_normalized, 2)
    signals.append(("fear_greed", fg_normalized, 0.30))

    fg_label = (
        "Extreme Fear" if fear_greed < 25
        else "Fear" if fear_greed < 45
        else "Extreme Greed" if fear_greed > 75
        else "Greed" if fear_greed > 55
        else "Neutral"
    )
    result["details"].append(f"Fear & Greed: {fg_label} ({fear_greed}/100)")

    # Source 2: Reddit sentiment
    try:
        reddit_score, reddit_details = _fetch_reddit_sentiment(symbol)
        result["reddit_score"] = round(reddit_score, 2)
        signals.append(("reddit", reddit_score, 0.35))
        result["details"].extend(reddit_details)
    except Exception as e:
        logger.debug(f"Reddit sentiment failed for {symbol}: {e}")

    # Source 3: CoinGecko price momentum
    try:
        momentum_score, momentum_details = _fetch_coingecko_momentum(symbol)
        result["momentum_score"] = round(momentum_score, 2)
        signals.append(("momentum", momentum_score, 0.35))
        result["details"].extend(momentum_details)
    except Exception as e:
        logger.debug(f"CoinGecko momentum failed for {symbol}: {e}")

    # Combine with weights
    if signals:
        total_weight = sum(w for _, _, w in signals)
        composite = sum(s * w for _, s, w in signals) / total_weight if total_weight > 0 else 0
        result["composite_score"] = round(max(-1.0, min(1.0, composite)), 3)

    _sentiment_cache[symbol] = {"data": result, "time": now}
    return result
