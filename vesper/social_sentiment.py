"""Social sentiment analysis — X/Twitter data for trade signal confirmation.

Uses Perplexity Sonar to search recent X/Twitter posts about a coin/stock,
then extracts sentiment signals to confirm or reject trading decisions.

No direct X API key needed — Perplexity searches social media as part of
its web search capability.
"""

import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Cache: {symbol: {"data": {...}, "ts": float}}
_social_cache: dict = {}
SOCIAL_CACHE_TTL = 1800  # 30 minutes


def _get_perplexity_key() -> str:
    """Get Perplexity API key from environment or data file."""
    import os
    import json
    key = os.environ.get("PERPLEXITY_API_KEY", "")
    keys_file = os.path.join(os.environ.get("VESPER_DATA_DIR", "data"), "api_keys.json")
    try:
        with open(keys_file) as f:
            stored = json.load(f)
        key = stored.get("PERPLEXITY_API_KEY") or key
    except (FileNotFoundError, ValueError):
        pass
    return key


def fetch_social_sentiment(symbol: str, asset_type: str = "crypto") -> Optional[dict]:
    """Fetch social sentiment for a symbol from X/Twitter via Perplexity search.

    Args:
        symbol: Trading symbol (e.g., "BTC/USDT", "AAPL/USD", "PRED:question")
        asset_type: "crypto", "stock", or "prediction"

    Returns:
        Dict with sentiment data or None if unavailable:
        {
            "symbol": str,
            "sentiment": "bullish" | "bearish" | "neutral" | "mixed",
            "confidence": float (0-1),
            "summary": str,  # Human-readable summary
            "key_posts": [str],  # Notable tweets/posts
            "influencer_sentiment": str,  # What major accounts are saying
            "volume_indicator": "high" | "normal" | "low",  # Social mention volume
            "ts": float,  # Timestamp
        }
    """
    # Check cache
    cache_key = symbol.strip().upper()
    cached = _social_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < SOCIAL_CACHE_TTL:
        return cached["data"]

    api_key = _get_perplexity_key()
    if not api_key:
        logger.warning("[social] No Perplexity API key — skipping social sentiment")
        return None

    # Build the search query based on asset type
    base_sym = symbol.split("/")[0].replace("PRED:", "")
    if asset_type == "crypto":
        query = (
            f"Search X (Twitter) for the latest posts and sentiment about {base_sym} cryptocurrency "
            f"in the last 24 hours. What are crypto influencers and traders saying? "
            f"Is the overall sentiment bullish, bearish, or neutral? "
            f"Are there any notable whale movements or upcoming catalysts being discussed? "
            f"Summarize the key tweets and overall social media mood."
        )
    elif asset_type == "stock":
        query = (
            f"Search X (Twitter) for the latest posts about ${base_sym} stock "
            f"in the last 24 hours. What are traders, analysts, and finance influencers saying? "
            f"Is sentiment bullish, bearish, or neutral? "
            f"Any earnings expectations, analyst upgrades/downgrades, or notable news being discussed?"
        )
    else:
        query = (
            f"Search X (Twitter) for the latest posts about: {base_sym}. "
            f"What is the social media consensus? Are people discussing this topic actively?"
        )

    try:
        resp = httpx.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a social media sentiment analyst. Analyze X/Twitter posts about "
                            "the given asset. Respond in this exact JSON format:\n"
                            '{"sentiment": "bullish|bearish|neutral|mixed", '
                            '"confidence": 0.0-1.0, '
                            '"summary": "2-3 sentence summary", '
                            '"key_posts": ["notable tweet 1", "notable tweet 2"], '
                            '"influencer_sentiment": "what major accounts say", '
                            '"volume_indicator": "high|normal|low"}'
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                "max_tokens": 800,
                "temperature": 0.1,
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]

        # Try to parse JSON from the response
        import json
        # Handle markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Extract sentiment from plain text
            lower = content.lower()
            sentiment = "neutral"
            if "bullish" in lower:
                sentiment = "bullish"
            elif "bearish" in lower:
                sentiment = "bearish"
            elif "mixed" in lower:
                sentiment = "mixed"
            data = {
                "sentiment": sentiment,
                "confidence": 0.5,
                "summary": content[:500],
                "key_posts": [],
                "influencer_sentiment": "",
                "volume_indicator": "normal",
            }

        result = {
            "symbol": symbol,
            "sentiment": data.get("sentiment", "neutral"),
            "confidence": min(1.0, max(0.0, float(data.get("confidence", 0.5)))),
            "summary": data.get("summary", ""),
            "key_posts": data.get("key_posts", [])[:5],
            "influencer_sentiment": data.get("influencer_sentiment", ""),
            "volume_indicator": data.get("volume_indicator", "normal"),
            "ts": time.time(),
        }

        # Log API usage
        try:
            from vesper.dashboard.database import log_api_usage
            log_api_usage("perplexity", "social_sentiment", 0, True)
        except Exception:
            pass

        # Cache the result
        _social_cache[cache_key] = {"data": result, "ts": time.time()}
        logger.info(f"[social] {symbol}: {result['sentiment']} (conf={result['confidence']:.1%})")
        return result

    except Exception as e:
        logger.error(f"[social] Failed to fetch sentiment for {symbol}: {e}")
        try:
            from vesper.dashboard.database import log_api_usage
            log_api_usage("perplexity", "social_sentiment", 0, False)
        except Exception:
            pass
        return None


def get_social_signal(symbol: str, asset_type: str = "crypto") -> dict:
    """Get a simplified social signal for trading decisions.

    Returns:
        {
            "social_score": float (-1 to +1, negative=bearish, positive=bullish),
            "social_volume": "high" | "normal" | "low",
            "social_summary": str,
            "should_boost": bool,  # True if social confirms the trade direction
        }
    """
    data = fetch_social_sentiment(symbol, asset_type)
    if not data:
        return {
            "social_score": 0,
            "social_volume": "unknown",
            "social_summary": "Social sentiment unavailable",
            "should_boost": False,
        }

    # Convert sentiment to score
    sentiment_scores = {
        "bullish": 0.7,
        "bearish": -0.7,
        "neutral": 0.0,
        "mixed": 0.1,
    }
    base_score = sentiment_scores.get(data["sentiment"], 0)
    score = base_score * data["confidence"]

    # Volume boost: high social volume = stronger signal
    volume_mult = {"high": 1.3, "normal": 1.0, "low": 0.7}
    score *= volume_mult.get(data["volume_indicator"], 1.0)
    score = max(-1.0, min(1.0, score))

    return {
        "social_score": round(score, 2),
        "social_volume": data["volume_indicator"],
        "social_summary": data["summary"],
        "should_boost": score > 0.3,  # Bullish with decent confidence
    }
