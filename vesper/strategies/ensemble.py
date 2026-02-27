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

    def __init__(self, min_confidence: float = 0.5):
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
    """Enhanced ensemble that uses multi-timeframe, order book, and sentiment data.

    Reads extra fields from the snapshot:
    - tf_alignment: 0.0-1.0 (multi-timeframe EMA alignment)
    - buy_pressure: 0.0-1.0 (order book bid/ask ratio)
    - fear_greed: 0-100 (crypto Fear & Greed Index)
    - adx: trend strength (>25 = strong trend)
    - adx_4h: 4h trend strength
    """

    name = "smart_auto"

    def analyze(self, snapshot: dict) -> StrategyResult:
        base = super().analyze(snapshot)

        # Extra signals from enhanced snapshot
        alignment = snapshot.get("tf_alignment", 0.5)
        buy_pressure = snapshot.get("buy_pressure", 0.5)
        fear_greed = snapshot.get("fear_greed", 50)
        adx = snapshot.get("adx", 0)
        adx_4h = snapshot.get("adx_4h", 0)

        confidence = base.confidence
        extra_reasons = []

        if base.signal in (Signal.BUY, Signal.SELL):
            # Modifier 1: Multi-TF alignment (+/- 0.15)
            if base.signal == Signal.BUY:
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

            # Modifier 2: Order book pressure (+/- 0.10)
            if base.signal == Signal.BUY:
                ob_boost = (buy_pressure - 0.5) * 0.2
            else:
                ob_boost = (0.5 - buy_pressure) * 0.2
            confidence += ob_boost
            if buy_pressure > 0.6:
                extra_reasons.append(f"order book bullish ({buy_pressure:.0%})")
            elif buy_pressure < 0.4:
                extra_reasons.append(f"order book bearish ({buy_pressure:.0%})")

            # Modifier 3: Fear/Greed contrarian (+/- 0.10)
            if base.signal == Signal.BUY and fear_greed < 25:
                confidence += 0.10
                extra_reasons.append(f"extreme fear ({fear_greed}) — contrarian buy")
            elif base.signal == Signal.BUY and fear_greed > 75:
                confidence -= 0.10
                extra_reasons.append(f"extreme greed ({fear_greed}) — caution")
            elif base.signal == Signal.SELL and fear_greed > 75:
                confidence += 0.10
                extra_reasons.append(f"extreme greed ({fear_greed}) — confirms sell")

            # Modifier 4: ADX trend strength (+/- 0.05)
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
            signal=base.signal,
            confidence=confidence,
            reason=reason,
            strategy_name=self.name,
        )
