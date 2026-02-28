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

from vesper.dashboard.database import log_api_usage

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

    usage = data.get("usage", {})
    in_tok = usage.get("prompt_tokens", 0)
    out_tok = usage.get("completion_tokens", 0)
    cost = in_tok * 1e-6 + out_tok * 1e-6  # Sonar: ~$1/M in, ~$1/M out
    log_api_usage("perplexity", "sonar", in_tok, out_tok, cost, "market_search")

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

    usage = data.get("usage", {})
    in_tok = usage.get("input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)
    cost = in_tok * 1e-6 + out_tok * 5e-6  # Haiku 4.5: $1/M in, $5/M out
    log_api_usage("anthropic", "claude-haiku-4-5", in_tok, out_tok, cost, "market_analyze")

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
    logger.info(f"[research_market] '{question[:60]}...' — keys: pplx={'yes' if pplx_key else 'NO'}, anthropic={'yes' if anthropic_key else 'NO'}")

    if not pplx_key:
        logger.warning("[research_market] No Perplexity API key, skipping")
        return _empty_result("No Perplexity API key available")

    try:
        # Stage 1: Perplexity web search
        logger.info(f"[research_market] Calling Perplexity search...")
        search = _perplexity_search(question, market_probability, category, pplx_key)
        logger.info(f"[research_market] Perplexity returned {len(search.get('search_results', ''))} chars")

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

    usage = data.get("usage", {})
    in_tok = usage.get("prompt_tokens", 0)
    out_tok = usage.get("completion_tokens", 0)
    cost = in_tok * 1e-6 + out_tok * 1e-6
    log_api_usage("perplexity", "sonar", in_tok, out_tok, cost, "market_analyze_fallback")

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


# ═══════════════════════════════════════
# Asset Deep Research (Crypto + Stocks)
# ═══════════════════════════════════════

# Separate cache for asset research
_asset_cache: dict = {}
ASSET_CACHE_TTL = 2 * 3600  # 2 hours


def research_asset(
    symbol: str,
    indicators: dict,
    price: float,
    asset_type: str = "crypto",
) -> dict:
    """Deep research on a specific crypto or stock asset.

    Stage 1: Perplexity searches for current news, events, catalysts
    Stage 2: Claude combines technical indicators + fundamental research

    Returns: {signal, confidence, reasoning, factors, sources, search_summary}
    """
    cache_key = symbol.lower().strip()
    cached = _asset_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < ASSET_CACHE_TTL:
        return cached["data"]

    pplx_key, anthropic_key = _get_platform_keys()
    logger.info(f"[research_asset] {symbol} @ ${price:.4f} — keys: pplx={'yes' if pplx_key else 'NO'}, anthropic={'yes' if anthropic_key else 'NO'}")
    if not pplx_key:
        logger.warning(f"[research_asset] {symbol}: no Perplexity API key, skipping")
        return _empty_asset_result("No Perplexity API key")

    ticker = symbol.split("/")[0]  # "BTC/USDT" → "BTC"

    try:
        # Stage 1: Perplexity web search for the asset
        logger.info(f"[research_asset] {symbol}: calling Perplexity search...")
        search = _perplexity_asset_search(ticker, price, asset_type, pplx_key)
        logger.info(f"[research_asset] {symbol}: Perplexity returned {len(search.get('search_results', ''))} chars, {len(search.get('citations', []))} citations")

        if not anthropic_key:
            logger.info(f"[research_asset] {symbol}: no Anthropic key, returning Perplexity-only result")
            result = _asset_result_from_search(search)
            _asset_cache[cache_key] = {"data": result, "ts": time.time()}
            return result

        # Stage 2: Claude combines technical + fundamental
        logger.info(f"[research_asset] {symbol}: calling Claude analysis...")
        analysis = _claude_asset_analyze(
            ticker, price, indicators, search["search_results"],
            asset_type, anthropic_key,
        )
        result = {
            "signal": analysis.get("signal", "HOLD"),
            "confidence": float(analysis.get("confidence", 0.5)),
            "reasoning": analysis.get("reasoning", ""),
            "bullish_factors": analysis.get("bullish_factors", []),
            "bearish_factors": analysis.get("bearish_factors", []),
            "catalysts": analysis.get("catalysts", []),
            "sources": search.get("citations", [])[:5],
            "search_summary": search["search_results"][:500],
            "researched": True,
            "engine": "perplexity+claude",
            "research_ts": time.time(),
        }
        _asset_cache[cache_key] = {"data": result, "ts": time.time()}
        # Prune cache
        if len(_asset_cache) > 100:
            oldest = sorted(_asset_cache.items(), key=lambda x: x[1]["ts"])
            for k, _ in oldest[:30]:
                _asset_cache.pop(k, None)
        return result

    except Exception as e:
        logger.warning(f"Asset research failed for {symbol}: {e}")
        return _empty_asset_result(str(e))


def _perplexity_asset_search(ticker: str, price: float, asset_type: str, api_key: str) -> dict:
    """Search the web for current information about a crypto/stock asset."""
    if asset_type == "stock":
        prompt = f"""Research the stock {ticker} thoroughly. Current price: ${price:.2f}.

Find:
- Latest earnings reports, revenue, guidance
- Analyst upgrades/downgrades and price targets from the last 7 days
- Major news: partnerships, product launches, regulatory actions
- Sector/industry trends affecting this stock
- Any upcoming catalysts (earnings dates, FDA decisions, product launches)
- Short interest and institutional activity

Provide a comprehensive, fact-based summary with specific numbers and dates."""
    else:
        prompt = f"""Research the cryptocurrency {ticker} thoroughly. Current price: ${price:.4f}.

Find:
- Latest protocol updates, upgrades, or technical milestones
- Major partnerships, integrations, or ecosystem developments
- Exchange listing/delisting news
- Whale accumulation or distribution patterns
- Regulatory developments affecting this asset
- Social media sentiment and community activity trends
- On-chain metrics trends (TVL, active addresses, transaction volume)
- Any upcoming catalysts (unlocks, halvings, airdrops)

Provide a comprehensive, fact-based summary with specific numbers and dates."""

    resp = httpx.post(
        PERPLEXITY_API,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "sonar",
            "messages": [
                {"role": "system", "content": "You are a financial research analyst. Provide thorough, factual summaries with specific details."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 1200,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    usage = data.get("usage", {})
    in_tok = usage.get("prompt_tokens", 0)
    out_tok = usage.get("completion_tokens", 0)
    cost = in_tok * 1e-6 + out_tok * 1e-6
    log_api_usage("perplexity", "sonar", in_tok, out_tok, cost, "asset_search")

    return {
        "search_results": data["choices"][0]["message"]["content"].strip(),
        "citations": data.get("citations", []),
    }


def _claude_asset_analyze(
    ticker: str, price: float, indicators: dict,
    search_results: str, asset_type: str, api_key: str,
) -> dict:
    """Claude combines technical indicators + fundamental research into a trading signal."""
    # Format indicators
    ind_lines = []
    for k, v in indicators.items():
        if isinstance(v, float):
            ind_lines.append(f"  {k}: {v:.4f}")
        else:
            ind_lines.append(f"  {k}: {v}")
    ind_text = "\n".join(ind_lines) if ind_lines else "  (none available)"

    prompt = f"""You are an expert {asset_type} trader and analyst. Combine the technical indicators AND fundamental research below to produce a trading signal.

**ASSET:** {ticker} @ ${price:.4f}

**TECHNICAL INDICATORS:**
{ind_text}

**FUNDAMENTAL RESEARCH (from web search):**
{search_results}

INSTRUCTIONS:
1. Weigh both technical and fundamental factors
2. Technical indicators show SHORT-TERM momentum and trends
3. Fundamental research shows MEDIUM-TERM catalysts and sentiment
4. If fundamentals and technicals agree → high confidence
5. If they disagree → lower confidence, lean toward fundamentals for direction

Respond in this exact JSON format only (no markdown):
{{"signal": "BUY", "confidence": 0.75, "reasoning": "2-3 sentence synthesis", "bullish_factors": ["specific factor 1", "specific factor 2"], "bearish_factors": ["specific factor 1", "specific factor 2"], "catalysts": ["upcoming event or catalyst"]}}

Rules:
- signal: "BUY", "SELL", or "HOLD"
- confidence: float 0.0-1.0
- reasoning: concise synthesis of why this signal
- bullish_factors: 2-4 evidence-based reasons for upside
- bearish_factors: 2-4 evidence-based reasons for downside
- catalysts: 1-3 upcoming events that could move the price"""

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
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    usage = data.get("usage", {})
    in_tok = usage.get("input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)
    cost = in_tok * 1e-6 + out_tok * 5e-6
    log_api_usage("anthropic", "claude-haiku-4-5", in_tok, out_tok, cost, "asset_analyze")

    content = data["content"][0]["text"].strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
    return json.loads(content)


def _asset_result_from_search(search: dict) -> dict:
    """Fallback when Claude is unavailable — return raw search without analysis."""
    return {
        "signal": "HOLD",
        "confidence": 0.3,
        "reasoning": "Deep analysis unavailable, search results only",
        "bullish_factors": [],
        "bearish_factors": [],
        "catalysts": [],
        "sources": search.get("citations", [])[:5],
        "search_summary": search["search_results"][:500],
        "researched": True,
        "engine": "perplexity",
        "research_ts": time.time(),
    }


def _empty_asset_result(reason: str = "") -> dict:
    return {
        "signal": "HOLD",
        "confidence": 0.0,
        "reasoning": reason,
        "bullish_factors": [],
        "bearish_factors": [],
        "catalysts": [],
        "sources": [],
        "search_summary": "",
        "researched": False,
        "engine": "none",
        "research_ts": 0,
    }
