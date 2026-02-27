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
