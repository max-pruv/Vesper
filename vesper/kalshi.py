"""Kalshi integration — fetch prediction markets via public REST API.

Kalshi is a CFTC-regulated prediction market (powers Coinbase predictions).
Public market data requires no authentication.
"""

import json
from datetime import datetime, timezone, timedelta

import httpx

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"


def fetch_events(limit: int = 50, status: str = "open") -> list[dict]:
    """Fetch events with nested markets from Kalshi."""
    resp = httpx.get(
        f"{KALSHI_API}/events",
        params={
            "limit": str(limit),
            "status": status,
            "with_nested_markets": "true",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("events", [])


def parse_kalshi_market(market: dict, event: dict) -> dict:
    """Convert a Kalshi market to our unified format."""
    yes_bid = market.get("yes_bid", 0) or 0
    yes_ask = market.get("yes_ask", 0) or 0
    no_bid = market.get("no_bid", 0) or 0
    no_ask = market.get("no_ask", 0) or 0

    # Kalshi prices are in cents (0-100), convert to 0-1 for compatibility
    yes_mid = ((yes_bid + yes_ask) / 2) / 100 if (yes_bid + yes_ask) > 0 else 0
    no_mid = ((no_bid + no_ask) / 2) / 100 if (no_bid + no_ask) > 0 else 0

    # If no YES price, derive from NO
    if yes_mid == 0 and no_mid > 0:
        yes_mid = 1 - no_mid

    spread = abs(yes_ask - yes_bid) / 100 if yes_ask > 0 else 0

    outcomes = [
        {
            "name": "Yes",
            "price": yes_mid,
            "probability": round(yes_mid * 100, 1),
            "token_id": market.get("ticker", ""),
        },
        {
            "name": "No",
            "price": 1 - yes_mid if yes_mid > 0 else 0,
            "probability": round((1 - yes_mid) * 100, 1) if yes_mid > 0 else 0,
            "token_id": "",
        },
    ]

    return {
        "id": market.get("ticker", ""),
        "question": market.get("title", ""),
        "slug": market.get("ticker", ""),
        "outcomes": outcomes,
        "volume": market.get("volume", 0) or 0,
        "volume_24h": market.get("volume_24h", 0) or 0,
        "liquidity": market.get("open_interest", 0) or 0,
        "end_date": market.get("expected_expiration_time") or market.get("close_time"),
        "active": market.get("status") == "active",
        "closed": market.get("status") in ("closed", "settled"),
        "best_bid": yes_bid / 100,
        "best_ask": yes_ask / 100,
        "spread": spread,
        "source": "kalshi",
        "category": event.get("category", ""),
        "event_title": event.get("title", ""),
        "event_slug": event.get("event_ticker", ""),
        "rules": market.get("rules_primary", ""),
    }


def _days_until(end_date_str: str | None) -> float:
    """Return days until end date, or 999 if missing."""
    if not end_date_str:
        return 999
    try:
        end = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return (end - now).total_seconds() / 86400
    except (ValueError, TypeError):
        return 999


def get_kalshi_markets(limit: int = 20, max_days: int = 0) -> list[dict]:
    """Get active Kalshi prediction markets with AI edge estimation.

    Args:
        limit: Maximum number of markets to return.
        max_days: Only include markets expiring within N days (0=all).

    Returns parsed markets sorted by volume, with AI edge estimates.
    """
    from vesper.polymarket import estimate_true_probability, calculate_edge

    events = fetch_events(limit=50, status="open")
    markets = []

    for event in events:
        for raw_market in event.get("markets", []):
            if raw_market.get("status") in ("closed", "settled"):
                continue
            # Skip multivariate combo bets
            if raw_market.get("mve_collection_ticker"):
                continue
            # Skip zero-volume markets
            if (raw_market.get("volume", 0) or 0) < 10:
                continue

            parsed = parse_kalshi_market(raw_market, event)

            days_left = _days_until(parsed.get("end_date"))
            if days_left < 0.5:
                continue
            if max_days > 0 and days_left > max_days:
                continue

            parsed["days_left"] = round(days_left, 1)
            parsed["tags"] = [event.get("category", "")]

            # AI edge estimation (reuses Polymarket's microstructure analysis)
            ai_data = estimate_true_probability(parsed)
            parsed["ai_probability"] = ai_data["ai_probability"]
            parsed["market_probability"] = ai_data["market_probability"]
            parsed["ai_edge"] = ai_data["edge"]
            parsed["ai_confidence"] = ai_data["confidence"]
            parsed["ai_signals"] = ai_data["signals"]

            edge = ai_data["edge"]
            if edge.get("side") and edge.get("ev_per_dollar", 0) > 0:
                parsed["potential_gain_100"] = round(edge["ev_per_dollar"] * 100, 2)
            else:
                parsed["potential_gain_100"] = 0

            markets.append(parsed)

    markets.sort(
        key=lambda m: (
            m.get("ai_edge", {}).get("ev_per_dollar", 0),
            m.get("volume_24h", 0),
        ),
        reverse=True,
    )
    return markets[:limit]

