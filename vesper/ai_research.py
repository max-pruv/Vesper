"""Deep AI research for prediction market probability estimation.

Two-stage pipeline:
  1. Perplexity Sonar — searches the web for current news, data, expert analyses
  2. Claude (Anthropic) — deep reasoning on the gathered evidence to produce
     a calibrated probability estimate with structured factors

Platform-provided API keys (env vars), no user configuration needed.
Results cached for 4 hours to minimize costs.
"""

import json
import logging
import os
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

PERPLEXITY_API = "https://api.perplexity.ai/chat/completions"
ANTHROPIC_API = "https://api.anthropic.com/v1/messages"

# In-memory cache: {question_hash: {"data": {...}, "ts": float}}
_research_cache: dict = {}
CACHE_TTL = 4 * 3600  # 4 hours


def _get_platform_keys() -> tuple[str, str]:
    """Get platform-provided API keys from environment."""
    return (
        os.environ.get("PERPLEXITY_API_KEY", ""),
        os.environ.get("ANTHROPIC_API_KEY", ""),
    )


def _cache_key(question: str) -> str:
    return question.strip().lower()


def _get_cached(question: str) -> Optional[dict]:
    key = _cache_key(question)
    entry = _research_cache.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL:
        return entry["data"]
    return None


def _set_cache(question: str, data: dict):
    key = _cache_key(question)
    _research_cache[key] = {"data": data, "ts": time.time()}
    if len(_research_cache) > 200:
        oldest = sorted(_research_cache.items(), key=lambda x: x[1]["ts"])
        for k, _ in oldest[:50]:
            _research_cache.pop(k, None)


# ── Stage 1: Perplexity web search ──

def _perplexity_search(question: str, market_probability: float, category: str, api_key: str) -> dict:
    """Search the web for relevant information about a prediction market question.

    Returns dict with: search_results (text), citations (list of URLs).
    """
    prompt = f"""Research this prediction market question thoroughly. Find the most recent and relevant information.

**Question:** {question}
**Current market price:** {market_probability:.1f}%
{f'**Category:** {category}' if category else ''}

Search for:
- Latest news and developments related to this question
- Expert opinions, forecasts, and analyses
- Relevant data points, statistics, and polls
- Historical context and base rates

Provide a comprehensive research summary with all relevant findings. Include specific facts, dates, numbers, and expert names where available."""

    resp = httpx.post(
        PERPLEXITY_API,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "sonar",
            "messages": [
                {"role": "system", "content": "You are a research assistant. Provide thorough, factual research summaries with specific details."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 1000,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    content = data["choices"][0]["message"]["content"].strip()
    citations = data.get("citations", [])

    return {
        "search_results": content,
        "citations": citations[:8],
    }


# ── Stage 2: Claude deep reasoning ──

def _claude_analyze(question: str, market_probability: float, search_results: str, api_key: str) -> dict:
    """Use Claude to analyze search results and produce a calibrated probability estimate.

    Returns dict with: probability, confidence, yes_factors, no_factors, reasoning.
    """
    prompt = f"""You are an expert prediction market analyst. Based on the research below, estimate the true probability for this question.

**QUESTION:** {question}
**CURRENT MARKET PRICE:** {market_probability:.1f}% (what the market currently implies)

**RESEARCH FINDINGS:**
{search_results}

INSTRUCTIONS:
1. Analyze all the evidence from the research findings
2. Consider base rates, recent developments, expert consensus, and key uncertainties
3. Provide a calibrated probability estimate (be specific, not vague)
4. If the evidence strongly supports one side, your probability should reflect that
5. If evidence is mixed or scarce, stay closer to the market price

Respond in this exact JSON format only (no markdown, no explanation outside the JSON):
{{"probability": 65, "confidence": "medium", "yes_factors": ["specific factor 1", "specific factor 2", "specific factor 3"], "no_factors": ["specific factor 1", "specific factor 2", "specific factor 3"], "reasoning": "2-3 sentence synthesis explaining your probability estimate based on the evidence"}}

Rules:
- probability: integer 1-99
- confidence: "low", "medium", or "high"
- yes_factors: 2-4 specific, evidence-based reasons supporting YES
- no_factors: 2-4 specific, evidence-based reasons supporting NO
- reasoning: concise synthesis grounded in the research findings"""

    resp = httpx.post(
        ANTHROPIC_API,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 600,
            "temperature": 0.1,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    content = data["content"][0]["text"].strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    return json.loads(content)


# ── Public API ──

def research_market(
    question: str,
    market_probability: float,
    api_key: str = "",
    category: str = "",
) -> dict:
    """Research a prediction market question using the Perplexity + Claude pipeline.

    Stage 1: Perplexity searches the web for relevant information
    Stage 2: Claude analyzes the evidence and produces a probability estimate

    Falls back to Perplexity-only if Anthropic key is unavailable.
    Uses platform-provided keys by default (env vars).

    Args:
        question: The market question
        market_probability: Current market-implied probability (0-100)
        api_key: Optional override for Perplexity key (uses env var if empty)
        category: Optional category for context

    Returns:
        Dict with: probability, confidence, yes_factors, no_factors,
        reasoning, sources, researched (bool)
    """
    cached = _get_cached(question)
    if cached:
        return cached

    pplx_key, anthropic_key = _get_platform_keys()
    # Allow per-user override (backwards compat), but prefer platform keys
    pplx_key = pplx_key or api_key

    if not pplx_key:
        return _empty_result("No Perplexity API key available")

    try:
        # Stage 1: Perplexity web search
        search = _perplexity_search(question, market_probability, category, pplx_key)

        # Stage 2: Claude deep reasoning (if available)
        if anthropic_key:
            try:
                analysis = _claude_analyze(
                    question, market_probability,
                    search["search_results"], anthropic_key,
                )
                research_data = {
                    "probability": max(1, min(99, int(analysis.get("probability", market_probability)))),
                    "confidence": analysis.get("confidence", "medium"),
                    "yes_factors": analysis.get("yes_factors", [])[:4],
                    "no_factors": analysis.get("no_factors", [])[:4],
                    "reasoning": analysis.get("reasoning", ""),
                    "sources": search["citations"][:5],
                    "researched": True,
                    "engine": "perplexity+claude",
                    "research_ts": time.time(),
                }
                _set_cache(question, research_data)
                return research_data
            except Exception as e:
                logger.warning(f"Claude analysis failed, falling back to Perplexity-only: {e}")

        # Fallback: Perplexity-only (search + synthesis in one call)
        return _perplexity_only(question, market_probability, category, pplx_key, search["citations"])

    except httpx.HTTPStatusError as e:
        logger.warning(f"Perplexity API error {e.response.status_code}: {e.response.text[:200]}")
        return _empty_result(f"API error: {e.response.status_code}")
    except Exception as e:
        logger.warning(f"Research failed: {e}")
        return _empty_result(str(e))


def _perplexity_only(
    question: str, market_probability: float,
    category: str, api_key: str, citations: list,
) -> dict:
    """Fallback: use Perplexity for both search and analysis."""
    prompt = f"""You are a prediction market analyst. Analyze this question and estimate the true probability.

**Question:** {question}
**Current market price:** {market_probability:.1f}%
{f'**Category:** {category}' if category else ''}

Research thoroughly, then respond in this exact JSON format (no markdown):
{{"probability": 65, "confidence": "medium", "yes_factors": ["factor 1", "factor 2"], "no_factors": ["factor 1", "factor 2"], "reasoning": "2-3 sentence summary"}}"""

    resp = httpx.post(
        PERPLEXITY_API,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "sonar",
            "messages": [
                {"role": "system", "content": "You are a prediction market research analyst. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 500,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    content = data["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    result = json.loads(content)
    all_citations = citations + data.get("citations", [])
    # Deduplicate citations
    seen = set()
    unique_citations = []
    for c in all_citations:
        if c not in seen:
            seen.add(c)
            unique_citations.append(c)

    research_data = {
        "probability": max(1, min(99, int(result.get("probability", market_probability)))),
        "confidence": result.get("confidence", "low"),
        "yes_factors": result.get("yes_factors", [])[:4],
        "no_factors": result.get("no_factors", [])[:4],
        "reasoning": result.get("reasoning", ""),
        "sources": unique_citations[:5],
        "researched": True,
        "engine": "perplexity",
        "research_ts": time.time(),
    }
    _set_cache(question, research_data)
    return research_data


def _empty_result(reason: str = "") -> dict:
    return {
        "probability": 0,
        "confidence": "none",
        "yes_factors": [],
        "no_factors": [],
        "reasoning": reason,
        "sources": [],
        "researched": False,
        "engine": "none",
        "research_ts": 0,
    }
