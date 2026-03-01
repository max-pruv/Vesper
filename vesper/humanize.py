"""Humanize AI signals and autopilot log entries for end-user readability."""

from __future__ import annotations


def humanize_signal(symbol: str, signal: str, confidence: float,
                    reason: str, snapshot: dict | None = None) -> str:
    """Transform a raw AI signal into a natural-language summary.

    Args:
        symbol: e.g. "BTC/USDT"
        signal: "BUY", "SELL", or "HOLD"
        confidence: 0.0 to 1.0
        reason: raw reason string from strategy
        snapshot: market snapshot dict (optional, for enrichment)

    Returns:
        A human-readable 1–2 sentence summary.
    """
    name = symbol.split("/")[0]
    conf_pct = int(confidence * 100)

    # Parse key indicators from the raw reason
    indicators = _parse_reason(reason)

    if signal == "BUY":
        return _build_buy_narrative(name, conf_pct, indicators, snapshot)
    elif signal == "SELL":
        return _build_sell_narrative(name, conf_pct, indicators, snapshot)
    elif signal == "SHORT":
        return _build_short_narrative(name, conf_pct, indicators, snapshot)
    else:
        return _build_hold_narrative(name, conf_pct, indicators, snapshot)


def humanize_autopilot_log(entry: dict) -> str:
    """Transform an autopilot log entry into a human-readable string.

    Args:
        entry: dict from autopilot_log in portfolio JSON

    Returns:
        A natural-language summary of what the autopilot did.
    """
    status = entry.get("status", "")
    scanned = entry.get("symbols_scanned", 0)
    positions = entry.get("positions", 0)
    max_pos = entry.get("max_positions", 3)
    deployed = entry.get("deployed_usd", 0)
    available = entry.get("available_usd", 0)
    actions = entry.get("actions", [])
    candidates = entry.get("candidates", [])
    top_signals = entry.get("top_signals", [])

    if status == "fully_deployed":
        return (
            f"All {max_pos} slots are active — the autopilot is fully deployed "
            f"with ${deployed:,.0f} in the market. Waiting for a position to close "
            f"before scanning for new opportunities."
        )

    if status == "low_funds":
        return (
            f"Not enough funds to open a new position (${available:,.0f} available). "
            f"Currently running {positions}/{max_pos} positions with ${deployed:,.0f} deployed."
        )

    if status == "no_opportunity" and not actions:
        # Describe what was scanned and the best signal found
        if top_signals:
            best = top_signals[0]
            best_name = best["symbol"].split("/")[0]
            best_conf = int(best["confidence"] * 1000) / 10  # e.g. 0.421 -> 42.1
            best_sig = best["signal"]
            return (
                f"Scanned {scanned} assets — no strong buy signal found. "
                f"Best candidate was {best_name} ({best_sig} at {best_conf}% confidence), "
                f"below the 55% threshold. ${available:,.0f} ready to deploy."
            )
        return (
            f"Scanned {scanned} assets but found no opportunity meeting the criteria. "
            f"${available:,.0f} standing by for the next scan."
        )

    if actions:
        # Describe what was bought
        parts = []
        for a in actions:
            sym_name = a.get("symbol", "?").split("/")[0]
            amt = a.get("amount_usd", 0)
            price = a.get("price", 0)
            conf = int(a.get("confidence", 0) * 100)
            whale = a.get("whale", 0)
            sentiment = a.get("sentiment", 0)

            extras = []
            if whale > 0.3:
                extras.append("whale activity detected")
            elif whale < -0.3:
                extras.append("whale selling pressure")
            if sentiment > 0.3:
                extras.append("positive market sentiment")
            elif sentiment < -0.3:
                extras.append("negative sentiment")

            reason_parts = f"at {conf}% AI confidence"
            if extras:
                reason_parts += f" ({', '.join(extras)})"

            parts.append(
                f"Bought ${amt:,.0f} of {sym_name} at ${price:,.2f} — {reason_parts}"
            )

        summary = ". ".join(parts) + "."
        remaining = f" {len(candidates) - len(actions)} other candidates were considered." if len(candidates) > len(actions) else ""
        return f"Scanned {scanned} assets. {summary}{remaining}"

    return f"Autopilot scan completed — {scanned} assets checked, {positions}/{max_pos} slots active."


def humanize_reasoning(details: list[str], action: str, summary: str) -> str:
    """Transform AI reasoning details into a natural-language paragraph.

    Args:
        details: list of technical detail strings
        action: "HOLD", "SELL", etc.
        summary: raw technical summary string

    Returns:
        A human-friendly paragraph explaining the AI reasoning.
    """
    parts = []

    # Extract and translate each detail
    for d in details:
        d_lower = d.lower()

        # Price action
        if d_lower.startswith("price $"):
            parts.append(d)  # Already somewhat readable

        # RSI
        elif "rsi" in d_lower:
            if "oversold" in d_lower:
                parts.append("RSI indicates the asset is oversold — potential bounce ahead")
            elif "overbought" in d_lower:
                parts.append("RSI signals overbought conditions — watch for a pullback")
            elif "neutral" in d_lower:
                parts.append("RSI is in neutral territory")

        # EMA
        elif "ema" in d_lower:
            if "bullish" in d_lower:
                parts.append("Short-term momentum is bullish (fast EMA above slow EMA)")
            else:
                parts.append("Short-term momentum is bearish (fast EMA below slow EMA)")

        # MACD
        elif "macd" in d_lower:
            if "bullish" in d_lower:
                parts.append("MACD confirms upward momentum")
            else:
                parts.append("MACD signals downward momentum")

        # Multi-TF
        elif "multi-tf" in d_lower:
            if "aligned bullish" in d_lower:
                parts.append("Both short and long timeframes agree — bullish")
            elif "aligned bearish" in d_lower:
                parts.append("Both timeframes aligned bearish")
            else:
                parts.append("Mixed signals across timeframes")

        # Order book
        elif "order book" in d_lower:
            if "buy pressure" in d_lower:
                parts.append("Buyers dominate the order book")
            elif "sell pressure" in d_lower:
                parts.append("Sellers dominate the order book")

        # Fear & Greed
        elif "fear" in d_lower or "greed" in d_lower:
            if "extreme fear" in d_lower:
                parts.append("Market is in extreme fear — contrarian buy opportunity")
            elif "fear" in d_lower and "greed" not in d_lower:
                parts.append("Market sentiment is fearful")
            elif "extreme greed" in d_lower:
                parts.append("Market is in extreme greed — caution advised")
            elif "greed" in d_lower:
                parts.append("Market sentiment is greedy")
            else:
                parts.append(d)

        # Whale activity
        elif "whale" in d_lower:
            if "bullish" in d_lower or "accumulating" in d_lower:
                parts.append("Whales are accumulating — smart money is buying")
            elif "bearish" in d_lower or "distributing" in d_lower:
                parts.append("Whale distribution detected — big players selling")
            else:
                parts.append(d)

        # RSI divergence
        elif "rsi divergence" in d_lower:
            if "bullish" in d_lower:
                parts.append("Bullish RSI divergence — price dipping but momentum building")
            elif "bearish" in d_lower:
                parts.append("Bearish RSI divergence — price rising but momentum fading")
            else:
                parts.append(d)

        # Support/Resistance
        elif "support" in d_lower or "resistance" in d_lower:
            if "near support" in d_lower:
                parts.append("Price is near a key support level — good entry zone")
            elif "near resistance" in d_lower:
                parts.append("Price is testing resistance — potential rejection point")
            elif "approaching" in d_lower:
                parts.append(d)
            else:
                parts.append(d)

        # Volume profile / POC
        elif "poc" in d_lower or "volume profile" in d_lower or "value zone" in d_lower:
            if "below poc" in d_lower or "value zone buy" in d_lower:
                parts.append("Trading below fair value (Volume POC) — good entry area")
            elif "above poc" in d_lower and "overextended" in d_lower:
                parts.append("Price above fair value — potential mean reversion ahead")
            else:
                parts.append(d)

        # Trailing stop
        elif "trailing" in d_lower:
            parts.append(d)  # Already readable

        # Catch-all
        else:
            parts.append(d)

    if not parts:
        return summary or "AI is monitoring this position."

    return " · ".join(parts[:5])


# ─── Internal helpers ───

def _parse_reason(reason: str) -> dict:
    """Extract key signals from the raw reason string."""
    r = reason.lower()
    indicators = {
        "uptrend": "uptrend" in r or "ema12 > ema26" in r or "ema_12 > ema_26" in r,
        "downtrend": "downtrend" in r or "ema12 < ema26" in r or "ema_12 < ema_26" in r,
        "above_sma50": "above sma50" in r or "price above" in r,
        "below_sma50": "below sma50" in r or "price below" in r,
        "rsi_oversold": "oversold" in r,
        "rsi_overbought": "overbought" in r,
        "macd_bullish": ("macd" in r and "bullish" in r) or "macd crossover" in r,
        "macd_bearish": "macd" in r and "bearish" in r,
        "mean_reversion_buy": "mean reversion" in r and ("buy" in r or "oversold" in r),
        "mean_reversion_sell": "mean reversion" in r and ("sell" in r or "overbought" in r),
        "momentum_up": "momentum" in r and ("buy" in r or "bullish" in r),
        "momentum_down": "momentum" in r and ("sell" in r or "bearish" in r),
        "whale_bullish": "whale" in r and ("bullish" in r or "accumul" in r),
        "whale_bearish": "whale" in r and ("bearish" in r or "distribut" in r),
        "sentiment_positive": "sentiment" in r and ("positive" in r or "bullish" in r or "greed" in r),
        "sentiment_negative": "sentiment" in r and ("negative" in r or "bearish" in r or "fear" in r),
        "multi_tf_aligned": "multi" in r and "aligned" in r,
        "rsi_divergence_bull": "bullish rsi divergence" in r,
        "rsi_divergence_bear": "bearish rsi divergence" in r,
        "near_support": "near support" in r or "approaching support" in r,
        "near_resistance": "near resistance" in r or "approaching resistance" in r,
        "below_poc": "below poc" in r or "value zone buy" in r,
        "above_poc": "above poc" in r and ("overextended" in r or "less value" in r),
        "short_signal": "short signal" in r or "short" in r,
    }
    return indicators


def _build_buy_narrative(name: str, conf: int, ind: dict, snap: dict | None) -> str:
    """Build a natural-language BUY narrative."""
    strengths = []

    if ind["uptrend"]:
        strengths.append("strong uptrend confirmed")
    if ind["above_sma50"]:
        strengths.append("trading above key support")
    if ind["rsi_oversold"]:
        strengths.append("RSI shows oversold bounce opportunity")
    if ind["macd_bullish"]:
        strengths.append("MACD momentum turning up")
    if ind["mean_reversion_buy"]:
        strengths.append("price stretched below average — bounce expected")
    if ind["momentum_up"]:
        strengths.append("strong upward momentum")
    if ind["whale_bullish"]:
        strengths.append("whales are accumulating")
    if ind["sentiment_positive"]:
        strengths.append("market sentiment is bullish")
    if ind["multi_tf_aligned"]:
        strengths.append("multiple timeframes aligned")
    if ind["rsi_divergence_bull"]:
        strengths.append("RSI divergence signals a reversal")
    if ind["near_support"]:
        strengths.append("price near key support level")
    if ind["below_poc"]:
        strengths.append("trading in a high-value zone")

    if not strengths:
        strengths.append("technical indicators are favorable")

    # Pick top 2-3 for readability
    top = strengths[:3]
    narrative = f"{name} looks bullish — {', '.join(top)}."
    narrative += f" AI confidence: {conf}%."
    return narrative


def _build_sell_narrative(name: str, conf: int, ind: dict, snap: dict | None) -> str:
    """Build a natural-language SELL narrative."""
    concerns = []

    if ind["downtrend"]:
        concerns.append("downtrend in progress")
    if ind["below_sma50"]:
        concerns.append("trading below key support")
    if ind["rsi_overbought"]:
        concerns.append("RSI signals overbought conditions")
    if ind["macd_bearish"]:
        concerns.append("MACD momentum fading")
    if ind["mean_reversion_sell"]:
        concerns.append("price stretched above average — pullback likely")
    if ind["momentum_down"]:
        concerns.append("losing momentum")
    if ind["whale_bearish"]:
        concerns.append("whales are distributing")
    if ind["sentiment_negative"]:
        concerns.append("market sentiment is bearish")
    if ind["rsi_divergence_bear"]:
        concerns.append("RSI divergence confirms weakness")
    if ind["near_resistance"]:
        concerns.append("rejected at key resistance level")
    if ind["above_poc"]:
        concerns.append("price overextended above value zone")

    if not concerns:
        concerns.append("technical indicators suggest caution")

    top = concerns[:3]
    narrative = f"{name} looks bearish — {', '.join(top)}."
    narrative += f" AI confidence: {conf}%."
    return narrative


def _build_short_narrative(name: str, conf: int, ind: dict, snap: dict | None) -> str:
    """Build a natural-language SHORT narrative."""
    reasons = []

    if ind["downtrend"]:
        reasons.append("clear downtrend in progress")
    if ind["whale_bearish"]:
        reasons.append("whales are exiting aggressively")
    if ind["rsi_divergence_bear"]:
        reasons.append("bearish RSI divergence detected")
    if ind["near_resistance"]:
        reasons.append("rejected at resistance")
    if ind["macd_bearish"]:
        reasons.append("MACD confirms downward momentum")
    if ind["sentiment_negative"]:
        reasons.append("bearish market sentiment")
    if ind["above_poc"]:
        reasons.append("price above fair value")

    if not reasons:
        reasons.append("strong bearish conviction across multiple signals")

    top = reasons[:3]
    narrative = f"{name} short opportunity — {', '.join(top)}."
    narrative += f" AI confidence: {conf}%."
    return narrative


def _build_hold_narrative(name: str, conf: int, ind: dict, snap: dict | None) -> str:
    """Build a natural-language HOLD narrative."""
    observations = []

    if ind["uptrend"]:
        observations.append("trend is up but signals are mixed")
    elif ind["downtrend"]:
        observations.append("trend is down but no clear entry")
    else:
        observations.append("no clear directional bias")

    if ind["rsi_oversold"]:
        observations.append("RSI oversold but momentum hasn't confirmed")
    elif ind["rsi_overbought"]:
        observations.append("RSI overbought but no sell trigger yet")

    narrative = f"{name} is in a wait zone — {observations[0]}."
    narrative += f" Best to hold off for now."
    return narrative
