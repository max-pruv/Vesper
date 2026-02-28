"""Altcoin Hunter — autonomous altcoin trend detection and self-investing system.

Scans 50+ altcoins, scores each on multiple trend signals, dynamically allocates
capital to the strongest trends, and self-regulates with trailing stops and
auto-rebalancing.

Key features:
- Multi-factor trend scoring (momentum, RSI, EMA alignment, volume, ADX, BTC strength)
- Dynamic position sizing (more capital to stronger trends)
- Auto-rebalancing: exits weakening trends, enters new strong ones
- Trailing stops for profit protection
- Adaptive risk: tighter stops in volatile markets, wider in strong trends
"""

import logging
import time

from .base import Signal, Strategy, StrategyResult

logger = logging.getLogger(__name__)


def compute_trend_score(snapshot: dict, btc_snapshot: dict | None = None) -> dict:
    """Compute a multi-factor trend score for an altcoin.

    Returns dict with:
      - score: float (0.0 to 1.0, higher = stronger uptrend)
      - factors: dict of individual factor scores
      - signal: BUY, SELL, or HOLD
      - confidence: float (0.0 to 1.0)
    """
    factors = {}

    # ── Factor 1: Momentum (MACD) — weight 0.20 ──
    macd_hist = snapshot.get("macd_hist", 0)
    price = snapshot.get("price", 1)
    # Normalize MACD histogram as % of price
    macd_pct = (macd_hist / price) * 100 if price > 0 else 0
    # Score: positive histogram = bullish, scale to 0-1
    momentum_score = max(0, min(1, 0.5 + macd_pct * 10))
    factors["momentum"] = round(momentum_score, 3)

    # ── Factor 2: RSI Position — weight 0.15 ──
    rsi = snapshot.get("rsi", 50)
    if rsi > 70:
        # Overbought — slightly negative (but still trending)
        rsi_score = 0.6  # Still bullish zone
    elif rsi > 50:
        # Bullish zone — ideal for trend following
        rsi_score = 0.5 + (rsi - 50) / 40  # 0.5 to 1.0
    elif rsi > 30:
        # Bearish zone
        rsi_score = rsi / 60  # 0.5 to ~0.17
    else:
        # Oversold — potential reversal up
        rsi_score = 0.3
    factors["rsi"] = round(rsi_score, 3)

    # ── Factor 3: EMA Alignment — weight 0.20 ──
    ema_12 = snapshot.get("ema_12", 0)
    ema_26 = snapshot.get("ema_26", 0)
    sma_50 = snapshot.get("sma_50", 0)
    # Perfect alignment: price > EMA12 > EMA26 > SMA50
    alignment_points = 0
    if price > ema_12 > 0:
        alignment_points += 1
    if ema_12 > ema_26 > 0:
        alignment_points += 1
    if ema_26 > sma_50 > 0:
        alignment_points += 1
    if price > sma_50 > 0:
        alignment_points += 1
    ema_score = alignment_points / 4
    factors["ema_alignment"] = round(ema_score, 3)

    # ── Factor 4: Volume Surge — weight 0.15 ──
    volume = snapshot.get("volume", 0)
    vol_sma = snapshot.get("volume_sma", 0)
    if vol_sma > 0 and volume > 0:
        vol_ratio = volume / vol_sma
        # Above average volume = bullish confirmation
        vol_score = max(0, min(1, vol_ratio / 3))  # 1.5x avg → 0.5, 3x → 1.0
    else:
        vol_score = 0.5
    factors["volume_surge"] = round(vol_score, 3)

    # ── Factor 5: ADX Trend Strength — weight 0.15 ──
    adx = snapshot.get("adx", 0)
    if adx > 40:
        adx_score = 1.0  # Very strong trend
    elif adx > 25:
        adx_score = 0.7 + (adx - 25) / 50  # Strong trend
    elif adx > 15:
        adx_score = 0.3 + (adx - 15) / 25  # Developing trend
    else:
        adx_score = adx / 50  # Weak/no trend
    factors["adx_strength"] = round(adx_score, 3)

    # ── Factor 6: BTC Relative Strength — weight 0.15 ──
    btc_rsi_score = 0.5  # Neutral default
    if btc_snapshot:
        btc_rsi = btc_snapshot.get("rsi", 50)
        # If altcoin RSI > BTC RSI, it's outperforming
        rsi_diff = rsi - btc_rsi
        btc_rsi_score = max(0, min(1, 0.5 + rsi_diff / 40))
    factors["btc_relative"] = round(btc_rsi_score, 3)

    # ── Weighted Total ──
    weights = {
        "momentum": 0.20,
        "rsi": 0.15,
        "ema_alignment": 0.20,
        "volume_surge": 0.15,
        "adx_strength": 0.15,
        "btc_relative": 0.15,
    }
    total_score = sum(factors[k] * weights[k] for k in weights)

    # ── Derive Signal ──
    if total_score >= 0.55:
        sig = Signal.BUY
        confidence = min(1.0, (total_score - 0.4) * 1.67)
    elif total_score <= 0.35:
        sig = Signal.SELL
        confidence = min(1.0, (0.5 - total_score) * 2)
    else:
        sig = Signal.HOLD
        confidence = 0.0

    return {
        "score": round(total_score, 4),
        "factors": factors,
        "signal": sig,
        "confidence": round(confidence, 3),
    }


class AltcoinHunterStrategy(Strategy):
    """Multi-factor trend detection strategy for altcoin scanning.

    Uses 6 signal layers to score each altcoin's trend strength,
    then generates BUY/SELL/HOLD signals based on the composite score.
    """

    name = "altcoin_hunter"

    def __init__(self, btc_snapshot: dict | None = None):
        self.btc_snapshot = btc_snapshot

    def analyze(self, snapshot: dict) -> StrategyResult:
        result = compute_trend_score(snapshot, self.btc_snapshot)

        # Build a human-readable reason
        factors = result["factors"]
        top_factors = sorted(factors.items(), key=lambda x: abs(x[1] - 0.5), reverse=True)[:3]
        factor_strs = []
        for name, val in top_factors:
            direction = "bullish" if val > 0.5 else "bearish" if val < 0.5 else "neutral"
            factor_strs.append(f"{name}={val:.2f}({direction})")

        reason = (
            f"Trend score {result['score']:.2f} | "
            + ", ".join(factor_strs)
        )

        # Enrich with whale + sentiment if available
        whale = snapshot.get("whale_score", 0)
        sentiment = snapshot.get("sentiment_score", 0)
        if abs(whale) > 0.1:
            direction = "accumulating" if whale > 0 else "distributing"
            reason += f" | whales {direction}({whale:+.2f})"
            # Boost/reduce confidence based on whale activity
            if result["signal"] == Signal.BUY and whale > 0.2:
                result["confidence"] = min(1.0, result["confidence"] + 0.1)
            elif result["signal"] == Signal.BUY and whale < -0.3:
                result["confidence"] = max(0.0, result["confidence"] - 0.15)

        if abs(sentiment) > 0.1:
            direction = "bullish" if sentiment > 0 else "bearish"
            reason += f" | sentiment {direction}({sentiment:+.2f})"

        return StrategyResult(
            signal=result["signal"],
            confidence=result["confidence"],
            reason=reason,
            strategy_name=self.name,
        )
