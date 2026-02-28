"""Deep AI research for prediction market probability estimation.

Uses Perplexity Sonar API (web search + synthesis) to research each market
question and produce an evidence-based probability estimate with citations.

This replaces the shallow microstructure-only analysis with actual research:
- Searches recent news, expert analyses, data sources
- Synthesizes findings into a probability estimate
- Returns structured reasoning with citations
- Results cached for 4 hours to minimize API costs
"""

import json
import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

PERPLEXITY_API = "https://api.perplexity.ai/chat/completions"

# In-memory cache: {question_hash: {"data": {...}, "ts": float}}
_research_cache: dict = {}
CACHE_TTL = 4 * 3600  # 4 hours


def _cache_key(question: str) -> str:
    """Normalize question for cache lookup."""
    return question.strip().lower()


def _get_cached(question: str) -> Optional[dict]:
    """Return cached research if fresh, else None."""
    key = _cache_key(question)
    entry = _research_cache.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL:
        return entry["data"]
    return None


def _set_cache(question: str, data: dict):
    """Cache research result."""
    key = _cache_key(question)
    _research_cache[key] = {"data": data, "ts": time.time()}
    # Prune old entries (keep max 200)
    if len(_research_cache) > 200:
        oldest = sorted(_research_cache.items(), key=lambda x: x[1]["ts"])
        for k, _ in oldest[:50]:
            _research_cache.pop(k, None)


def research_market(
    question: str,
    market_probability: float,
    api_key: str,
    category: str = "",
) -> dict:
    """Research a prediction market question using Perplexity Sonar API.

    Args:
        question: The market question (e.g., "Will BTC hit $100K by March 2026?")
        market_probability: Current market-implied probability (0-100)
        api_key: Perplexity API key
        category: Optional category for context (e.g., "crypto", "politics")

    Returns:
        Dict with: probability, confidence, yes_factors, no_factors,
        reasoning, sources, researched (bool)
    """
    # Check cache first
    cached = _get_cached(question)
    if cached:
        return cached

    if not api_key:
        return _empty_result("No research API key configured")

    prompt = f"""You are a prediction market analyst. Analyze this prediction market question and estimate the true probability.

**Market question:** {question}
**Current market price:** {market_probability:.1f}% (implied probability)
{f'**Category:** {category}' if category else ''}

Research this thoroughly using recent news, data, expert analysis, and historical patterns. Consider:
- Recent relevant events and developments
- Expert opinions and forecasts
- Historical base rates for similar questions
- Key uncertainties and risk factors

Respond in this exact JSON format (no markdown, just raw JSON):
{{"probability": 65, "confidence": "medium", "yes_factors": ["factor 1", "factor 2"], "no_factors": ["factor 1", "factor 2"], "reasoning": "2-3 sentence summary of your analysis"}}

Rules:
- probability: your estimate 0-100 (integer)
- confidence: "low", "medium", or "high"
- yes_factors: 2-4 key reasons supporting YES (short, specific)
- no_factors: 2-4 key reasons supporting NO (short, specific)
- reasoning: brief synthesis of your research findings"""

    try:
        resp = httpx.post(
            PERPLEXITY_API,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": [
                    {"role": "system", "content": "You are a prediction market research analyst. Always respond with valid JSON only, no markdown formatting."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 500,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract the response text
        content = data["choices"][0]["message"]["content"].strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        # Parse JSON response
        result = json.loads(content)

        # Extract citations from Perplexity response
        citations = data.get("citations", [])

        research_data = {
            "probability": max(1, min(99, int(result.get("probability", market_probability)))),
            "confidence": result.get("confidence", "low"),
            "yes_factors": result.get("yes_factors", [])[:4],
            "no_factors": result.get("no_factors", [])[:4],
            "reasoning": result.get("reasoning", ""),
            "sources": citations[:5] if citations else [],
            "researched": True,
            "research_ts": time.time(),
        }

        _set_cache(question, research_data)
        return research_data

    except httpx.HTTPStatusError as e:
        logger.warning(f"Perplexity API error {e.response.status_code}: {e.response.text[:200]}")
        return _empty_result(f"API error: {e.response.status_code}")
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse Perplexity response as JSON: {content[:200]}")
        # Try to extract probability from free text
        return _empty_result("Could not parse research response")
    except Exception as e:
        logger.warning(f"Research failed: {e}")
        return _empty_result(str(e))


def _empty_result(reason: str = "") -> dict:
    """Return empty research result (fallback to microstructure)."""
    return {
        "probability": 0,
        "confidence": "none",
        "yes_factors": [],
        "no_factors": [],
        "reasoning": reason,
        "sources": [],
        "researched": False,
        "research_ts": 0,
    }
