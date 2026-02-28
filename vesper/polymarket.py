"""Polymarket integration — fetch prediction markets via Gamma API."""

import json
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


def get_trending_markets(limit: int = 20) -> list[dict]:
    """Get the highest-volume active prediction markets.

    Returns parsed markets sorted by 24h volume, with event context.
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
            parsed["event_title"] = event_title
            parsed["event_slug"] = event_slug
            parsed["tags"] = tags
            markets.append(parsed)

    # Sort by 24h volume (fallback to total volume)
    markets.sort(key=lambda m: (m["volume_24h"], m["volume"]), reverse=True)
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
    # If we buy YES at P, we win (1-P)/P fraction if correct, lose 1x if wrong
    ev_yes = true_probability * (1 / market_price - 1) - (1 - true_probability)

    # EV for buying NO at (1-market_price)
    no_price = 1 - market_price
    ev_no = (1 - true_probability) * (1 / no_price - 1) - true_probability

    if ev_yes > ev_no and ev_yes > 0:
        side = "YES"
        edge = (true_probability - market_price) / market_price * 100
        # Kelly criterion: f = (bp - q) / b where b = odds, p = true prob, q = 1-p
        b = (1 - market_price) / market_price  # decimal odds for YES
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

    kelly = max(0, min(kelly, 0.25))  # Cap at 25% of bankroll

    return {
        "edge_pct": round(edge, 1),
        "side": side,
        "kelly_pct": round(kelly * 100, 1),
        "ev_per_dollar": round(ev, 3),
    }
