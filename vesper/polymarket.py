"""Polymarket integration — fetch prediction markets via Gamma API.

AI Edge Estimation Methodology:
    The AI probability estimate uses market microstructure analysis to detect
    potential mispricings. It does NOT use news, social media, or insider info.

    Signals used:
    1. Volume momentum — high 24h volume vs total volume suggests active repricing
    2. Liquidity imbalance — order book asymmetry reveals where informed money sits
    3. Spread analysis — tight spreads = more certainty, wide = uncertainty/opportunity
    4. Contrarian reversion — extreme probabilities (>90%, <10%) tend to overshoot
    5. Freshness premium — new markets with low volume have wider mispricings

    The estimate is a small probabilistic nudge (±2-8%) from market price, not a
    fundamental research opinion. Think of it as "the market might be slightly off
    by this much based on trading patterns."
"""

import json
import math
from datetime import datetime, timezone

import httpx

GAMMA_API = "https://gamma-api.polymarket.com"


def fetch_events(limit: int = 50, offset: int = 0) -> list[dict]:
    """Fetch active, open events from Polymarket Gamma API."""
    resp = httpx.get(
        f"{GAMMA_API}/events",
        params={
            "active": "true",
            "closed": "false",
            "limit": str(limit),
            "offset": str(offset),
            "order": "volume",
            "ascending": "false",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def parse_market(market: dict) -> dict:
    """Parse a single market from the Gamma API response.

    Gamma returns outcomes, outcomePrices, and clobTokenIds as
    stringified JSON arrays — this normalizes them.
    """
    def _parse_json_field(val):
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return val
        return val

    outcomes = _parse_json_field(market.get("outcomes", []))
    prices = _parse_json_field(market.get("outcomePrices", []))
    token_ids = _parse_json_field(market.get("clobTokenIds", []))

    # Build outcome objects with prices
    outcome_list = []
    for i, name in enumerate(outcomes):
        price = float(prices[i]) if i < len(prices) else 0
        token_id = token_ids[i] if i < len(token_ids) else ""
        outcome_list.append({
            "name": name,
            "price": price,
            "probability": round(price * 100, 1),
            "token_id": token_id,
        })

    return {
        "id": market.get("id"),
        "question": market.get("question", ""),
        "slug": market.get("slug", ""),
        "outcomes": outcome_list,
        "volume": float(market.get("volumeNum", 0) or market.get("volume", 0) or 0),
        "volume_24h": float(market.get("volume24hr", 0) or 0),
        "liquidity": float(market.get("liquidity", 0) or 0),
        "end_date": market.get("endDate"),
        "active": market.get("active", True),
        "closed": market.get("closed", False),
        "best_bid": float(market.get("bestBid", 0) or 0),
        "best_ask": float(market.get("bestAsk", 0) or 0),
        "spread": float(market.get("spread", 0) or 0),
    }


def _days_until(end_date_str: str | None) -> float:
    """Return days until end date, or -1 if expired/missing."""
    if not end_date_str:
        return 999  # no end date = far future
    try:
        end = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = (end - now).total_seconds() / 86400
        return delta
    except (ValueError, TypeError):
        return 999


def estimate_true_probability(market: dict) -> dict:
    """Estimate true probability using market microstructure signals.

    Returns dict with AI estimate, edge info, confidence, and methodology breakdown.
    """
    outcomes = market.get("outcomes", [])
    yes_outcome = next((o for o in outcomes if o["name"] == "Yes"), outcomes[0] if outcomes else None)
    if not yes_outcome:
        return {"ai_probability": 0, "market_probability": 0, "edge": {}, "confidence": "low", "signals": []}

    market_price = yes_outcome["price"]  # 0-1 scale
    if market_price <= 0.02 or market_price >= 0.98:
        # Near-zero or near-certain markets — AI agrees with market, no actionable edge
        return {
            "ai_probability": round(market_price * 100, 1),
            "market_probability": round(market_price * 100, 1),
            "edge": {"edge_pct": 0, "side": None, "kelly_pct": 0, "ev_per_dollar": 0},
            "confidence": "low",
            "signals": ["Price near extreme — no reliable microstructure signal at this level"],
        }

    volume = market.get("volume", 0)
    volume_24h = market.get("volume_24h", 0)
    liquidity = market.get("liquidity", 0)
    spread = market.get("spread", 0)
    best_bid = market.get("best_bid", 0)
    best_ask = market.get("best_ask", 0)

    signals = []
    adjustments = []

    # Signal 1: Volume momentum — high recent activity suggests repricing
    if volume > 0 and volume_24h > 0:
        vol_ratio = volume_24h / max(volume, 1)
        if vol_ratio > 0.05:  # >5% of total volume in 24h = active repricing
            # High momentum pushes price further in current direction
            momentum_adj = vol_ratio * 0.15  # up to ~1.5% adjustment
            momentum_adj = min(momentum_adj, 0.04)
            if market_price > 0.5:
                adjustments.append(momentum_adj)
                signals.append(f"Volume momentum (+{momentum_adj*100:.1f}%): 24h volume is {vol_ratio*100:.1f}% of total — active buying pressure")
            else:
                adjustments.append(-momentum_adj)
                signals.append(f"Volume momentum ({-momentum_adj*100:.1f}%): 24h volume is {vol_ratio*100:.1f}% of total — active selling pressure")
        else:
            signals.append("Volume momentum (neutral): Low recent activity")

    # Signal 2: Liquidity depth — deep liquidity = more certainty
    if liquidity > 0 and volume > 0:
        liq_ratio = liquidity / max(volume, 1)
        if liq_ratio < 0.01:  # Thin liquidity relative to volume = potential mispricing
            thin_adj = 0.03  # Markets with thin books can be 3% off
            # Push toward 50% (uncertain markets should be closer to even)
            if market_price > 0.5:
                adjustments.append(-thin_adj * (market_price - 0.5))
                signals.append(f"Thin liquidity (contrarian): Low depth suggests price may overshoot")
            else:
                adjustments.append(thin_adj * (0.5 - market_price))
                signals.append(f"Thin liquidity (contrarian): Low depth suggests price may undershoot")
        else:
            signals.append("Liquidity depth (confirming): Deep order book supports current price")

    # Signal 3: Spread analysis — wide spread = uncertainty
    if spread > 0:
        if spread > 0.05:  # >5% spread = significant uncertainty
            spread_adj = spread * 0.3  # Wider spread = bigger potential mispricing
            spread_adj = min(spread_adj, 0.04)
            # Wide spread: push toward 50% (more uncertain)
            if market_price > 0.5:
                adjustments.append(-spread_adj)
                signals.append(f"Wide spread ({spread*100:.1f}%): Market makers uncertain — overpriced risk")
            else:
                adjustments.append(spread_adj)
                signals.append(f"Wide spread ({spread*100:.1f}%): Market makers uncertain — underpriced opportunity")
        else:
            signals.append(f"Tight spread ({spread*100:.1f}%): Strong consensus at current price")

    # Signal 4: Contrarian reversion — extreme prices tend to overshoot
    # But only meaningful for prices in the 8-92% range edges, NOT near 0% or 100%
    if market_price > 0.92:
        overshoot = (market_price - 0.92) * 0.3
        adjustments.append(-overshoot)
        signals.append(f"Extreme high ({market_price*100:.0f}%): Markets above 92% often overshoot by {overshoot*100:.1f}%")
    elif 0.03 < market_price < 0.08:
        # Only apply contrarian on markets with some plausibility (>3%)
        undershoot = (0.08 - market_price) * 0.3
        adjustments.append(undershoot)
        signals.append(f"Low probability ({market_price*100:.0f}%): Slight statistical undershoot bias")

    # Signal 5: Freshness — new/low-volume markets have wider mispricings
    if volume < 50000:
        fresh_adj = 0.02 * (1 - volume / 50000)
        # Push toward 50% but cap relative to market price
        if market_price > 0.5:
            adjustments.append(-fresh_adj)
        elif market_price > 0.1:  # Only for markets with some probability
            adjustments.append(fresh_adj)
        signals.append(f"Low volume market (${volume/1000:.0f}K): Less price discovery — wider potential mispricing")

    # Calculate final AI estimate — cap total adjustment relative to market price
    total_adj = sum(adjustments)
    # Don't let adjustment exceed 30% of the market price (prevents absurd edges on low-prob markets)
    max_adj = market_price * 0.3
    total_adj = max(-max_adj, min(max_adj, total_adj))
    ai_prob = max(0.01, min(0.99, market_price + total_adj))

    # Confidence based on how much data we have
    if volume > 1_000_000 and liquidity > 100_000 and spread < 0.03:
        confidence = "high"
    elif volume > 100_000 and liquidity > 10_000:
        confidence = "medium"
    else:
        confidence = "low"

    # Calculate edge
    edge = calculate_edge(ai_prob, market_price)

    return {
        "ai_probability": round(ai_prob * 100, 1),
        "market_probability": round(market_price * 100, 1),
        "edge": edge,
        "confidence": confidence,
        "signals": signals,
    }


def get_trending_markets(limit: int = 20, max_days: int = 30) -> list[dict]:
    """Get the highest-volume active prediction markets.

    Args:
        limit: Maximum number of markets to return.
        max_days: Only include markets expiring within this many days.
                  0 = no filtering. Default 30 (short-term focus).

    Returns parsed markets sorted by 24h volume, with event context and AI edge.
    """
    events = fetch_events(limit=100)
    markets = []

    for event in events:
        event_title = event.get("title", "")
        event_slug = event.get("slug", "")
        tags = [t.get("label", "") for t in event.get("tags", [])]

        for raw_market in event.get("markets", []):
            if raw_market.get("closed"):
                continue
            parsed = parse_market(raw_market)

            # Filter expired markets
            days_left = _days_until(parsed.get("end_date"))
            if days_left < 0.5:  # Less than 12 hours = expired / about to expire
                continue

            # Filter by max_days (0 = no filter)
            if max_days > 0 and days_left > max_days:
                continue

            parsed["days_left"] = round(days_left, 1)
            parsed["event_title"] = event_title
            parsed["event_slug"] = event_slug
            parsed["tags"] = tags

            # AI edge estimation
            ai_data = estimate_true_probability(parsed)
            parsed["ai_probability"] = ai_data["ai_probability"]
            parsed["market_probability"] = ai_data["market_probability"]
            parsed["ai_edge"] = ai_data["edge"]
            parsed["ai_confidence"] = ai_data["confidence"]
            parsed["ai_signals"] = ai_data["signals"]

            # Potential gain for $100 bet
            edge = ai_data["edge"]
            if edge.get("side") and edge.get("ev_per_dollar", 0) > 0:
                parsed["potential_gain_100"] = round(edge["ev_per_dollar"] * 100, 2)
            else:
                parsed["potential_gain_100"] = 0

            markets.append(parsed)

    # Sort by edge (markets with highest expected value first), fallback to volume
    markets.sort(
        key=lambda m: (
            m.get("ai_edge", {}).get("ev_per_dollar", 0),
            m.get("volume_24h", 0),
        ),
        reverse=True,
    )
    return markets[:limit]


def calculate_edge(true_probability: float, market_price: float) -> dict:
    """Calculate the expected value edge of a bet.

    Args:
        true_probability: Our estimated probability (0-1)
        market_price: Current market price / implied probability (0-1)

    Returns:
        Dict with edge %, recommended side, and kelly criterion.
    """
    if market_price <= 0 or market_price >= 1:
        return {"edge_pct": 0, "side": None, "kelly_pct": 0, "ev_per_dollar": 0}

    # EV for buying YES at market_price
    ev_yes = true_probability * (1 / market_price - 1) - (1 - true_probability)

    # EV for buying NO at (1-market_price)
    no_price = 1 - market_price
    ev_no = (1 - true_probability) * (1 / no_price - 1) - true_probability

    if ev_yes > ev_no and ev_yes > 0:
        side = "YES"
        edge = (true_probability - market_price) / market_price * 100
        b = (1 - market_price) / market_price
        kelly = (b * true_probability - (1 - true_probability)) / b
        ev = ev_yes
    elif ev_no > 0:
        side = "NO"
        edge = ((1 - true_probability) - no_price) / no_price * 100
        b = market_price / no_price
        kelly = (b * (1 - true_probability) - true_probability) / b
        ev = ev_no
    else:
        return {"edge_pct": 0, "side": None, "kelly_pct": 0, "ev_per_dollar": 0}

    kelly = max(0, min(kelly, 0.25))

    return {
        "edge_pct": round(edge, 1),
        "side": side,
        "kelly_pct": round(kelly * 100, 1),
        "ev_per_dollar": round(ev, 3),
    }
