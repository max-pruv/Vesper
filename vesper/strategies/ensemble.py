"""Ensemble strategy — combines multiple strategies with weighted voting."""

from .base import Signal, Strategy, StrategyResult
from .trend_following import TrendFollowingStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy


class EnsembleStrategy(Strategy):
    """
    Combines signals from multiple strategies using weighted voting.
    A trade is only executed when strategies agree with enough confidence.
    """

    name = "ensemble"

    def __init__(self, min_confidence: float = 0.35):
        self.min_confidence = min_confidence
        self.strategies: list[tuple[Strategy, float]] = [
            (TrendFollowingStrategy(), 0.35),
            (MeanReversionStrategy(), 0.30),
            (MomentumStrategy(), 0.35),
        ]

    def analyze(self, snapshot: dict) -> StrategyResult:
        results = []
        for strategy, weight in self.strategies:
            result = strategy.analyze(snapshot)
            results.append((result, weight))

        buy_score = 0.0
        sell_score = 0.0
        reasons = []

        for result, weight in results:
            weighted_confidence = result.confidence * weight
            if result.signal == Signal.BUY:
                buy_score += weighted_confidence
                reasons.append(f"[{result.strategy_name}] BUY({result.confidence:.2f}): {result.reason}")
            elif result.signal == Signal.SELL:
                sell_score += weighted_confidence
                reasons.append(f"[{result.strategy_name}] SELL({result.confidence:.2f}): {result.reason}")
            else:
                reasons.append(f"[{result.strategy_name}] HOLD: {result.reason}")

        combined_reason = " | ".join(reasons)

        if buy_score > sell_score and buy_score >= self.min_confidence:
            return StrategyResult(
                signal=Signal.BUY,
                confidence=min(buy_score, 1.0),
                reason=combined_reason,
                strategy_name=self.name,
            )

        if sell_score > buy_score and sell_score >= self.min_confidence:
            return StrategyResult(
                signal=Signal.SELL,
                confidence=min(sell_score, 1.0),
                reason=combined_reason,
                strategy_name=self.name,
            )

        return StrategyResult(
            signal=Signal.HOLD,
            confidence=0.0,
            reason=combined_reason,
            strategy_name=self.name,
        )


class EnhancedEnsemble(EnsembleStrategy):
    """AI Brain — combines technical analysis + whale tracking + sentiment.

    Uses 6 signal layers to decide entry AND exit:
    1. Technical ensemble (trend + mean-reversion + momentum)
    2. Multi-timeframe alignment (1h + 4h EMA agreement)
    3. Order book pressure (bid/ask imbalance)
    4. Whale activity (large trades, on-chain flows)
    5. Composite sentiment (Reddit + news + Fear&Greed + momentum)
    6. ADX trend strength

    The AI can override technical signals when whale + sentiment diverge.
    """

    name = "smart_auto"

    def analyze(self, snapshot: dict) -> StrategyResult:
        base = super().analyze(snapshot)

        # ── Layer 1: Technical base (from parent ensemble) ──
        alignment = snapshot.get("tf_alignment", 0.5)
        buy_pressure = snapshot.get("buy_pressure", 0.5)
        fear_greed = snapshot.get("fear_greed", 50)
        adx = snapshot.get("adx", 0)
        adx_4h = snapshot.get("adx_4h", 0)

        # ── Layer 4-5: Whale + Sentiment (NEW) ──
        whale_score = snapshot.get("whale_score", 0.0)     # -1 to +1
        sentiment_score = snapshot.get("sentiment_score", 0.0)  # -1 to +1

        confidence = base.confidence
        signal = base.signal
        extra_reasons = []

        # ── AI Override: Whale + Sentiment can flip HOLD → BUY or BUY → HOLD ──
        # If technicals say HOLD but whales are buying AND sentiment is bullish,
        # the AI can initiate a BUY (forward-looking edge)
        if signal == Signal.HOLD and whale_score > 0.3 and sentiment_score > 0.2:
            signal = Signal.BUY
            confidence = 0.50 + whale_score * 0.25 + sentiment_score * 0.25
            extra_reasons.append(
                f"AI override: whale accumulation ({whale_score:+.2f}) "
                f"+ bullish sentiment ({sentiment_score:+.2f}) → entering"
            )

        # If technicals say BUY but whales are dumping AND sentiment is bearish,
        # the AI blocks the entry
        elif signal == Signal.BUY and whale_score < -0.3 and sentiment_score < -0.1:
            signal = Signal.HOLD
            confidence = 0.0
            extra_reasons.append(
                f"AI override: whale distribution ({whale_score:+.2f}) "
                f"+ bearish sentiment ({sentiment_score:+.2f}) → blocking entry"
            )

        # If technicals say BUY but whales are heavily dumping, convert to SELL
        elif signal == Signal.BUY and whale_score < -0.5:
            signal = Signal.SELL
            confidence = abs(whale_score) * 0.5 + abs(sentiment_score) * 0.3
            extra_reasons.append(
                f"AI override: heavy whale selling ({whale_score:+.2f}) → exit signal"
            )

        if signal in (Signal.BUY, Signal.SELL):
            # ── Layer 2: Multi-TF alignment (±0.15) ──
            if signal == Signal.BUY:
                tf_boost = (alignment - 0.5) * 0.3
            else:
                tf_boost = (0.5 - alignment) * 0.3
            confidence += tf_boost
            if alignment == 1.0:
                extra_reasons.append("1h+4h aligned bullish")
            elif alignment == 0.0:
                extra_reasons.append("1h+4h aligned bearish")
            else:
                extra_reasons.append("timeframes mixed")

            # ── Layer 3: Order book pressure (±0.10) ──
            if signal == Signal.BUY:
                ob_boost = (buy_pressure - 0.5) * 0.2
            else:
                ob_boost = (0.5 - buy_pressure) * 0.2
            confidence += ob_boost
            if buy_pressure > 0.6:
                extra_reasons.append(f"order book bullish ({buy_pressure:.0%})")
            elif buy_pressure < 0.4:
                extra_reasons.append(f"order book bearish ({buy_pressure:.0%})")

            # ── Layer 4: Whale activity (±0.20 — high weight, forward-looking) ──
            if signal == Signal.BUY:
                whale_boost = whale_score * 0.20
            else:
                whale_boost = -whale_score * 0.20
            confidence += whale_boost
            if abs(whale_score) > 0.1:
                direction = "bullish" if whale_score > 0 else "bearish"
                extra_reasons.append(f"whale activity {direction} ({whale_score:+.2f})")

            # ── Layer 5: Composite sentiment (±0.15) ──
            if signal == Signal.BUY:
                sent_boost = sentiment_score * 0.15
            else:
                sent_boost = -sentiment_score * 0.15
            confidence += sent_boost
            if abs(sentiment_score) > 0.1:
                direction = "bullish" if sentiment_score > 0 else "bearish"
                extra_reasons.append(f"sentiment {direction} ({sentiment_score:+.2f})")

            # ── Layer 5b: Contrarian Fear/Greed (±0.10) ──
            if signal == Signal.BUY and fear_greed < 25:
                confidence += 0.10
                extra_reasons.append(f"extreme fear ({fear_greed}) — contrarian buy")
            elif signal == Signal.BUY and fear_greed > 75:
                confidence -= 0.10
                extra_reasons.append(f"extreme greed ({fear_greed}) — caution")
            elif signal == Signal.SELL and fear_greed > 75:
                confidence += 0.10
                extra_reasons.append(f"extreme greed ({fear_greed}) — confirms sell")

            # ── Layer 6: ADX trend strength (+0.05) ──
            strong_trend = adx > 25 or adx_4h > 25
            if strong_trend:
                confidence += 0.05
                extra_reasons.append(f"strong trend (ADX {adx:.0f})")

        confidence = max(0.0, min(confidence, 1.0))

        # Build enhanced reason
        reason = base.reason
        if extra_reasons:
            reason += " | " + ", ".join(extra_reasons)

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            reason=reason,
            strategy_name=self.name,
        )
