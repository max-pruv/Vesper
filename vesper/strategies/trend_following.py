"""Trend following strategy — EMA crossover with trend confirmation."""

from .base import Signal, Strategy, StrategyResult


class TrendFollowingStrategy(Strategy):
    """
    Uses EMA 12/26 crossover to detect trends.
    - BUY when EMA12 crosses above EMA26 and price is above SMA50
    - SELL when EMA12 crosses below EMA26 or price drops below SMA50
    """

    name = "trend_following"

    def analyze(self, snapshot: dict) -> StrategyResult:
        df = snapshot["df"]
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        ema_12 = latest["ema_12"]
        ema_26 = latest["ema_26"]
        prev_ema_12 = prev["ema_12"]
        prev_ema_26 = prev["ema_26"]
        price = latest["close"]
        sma_50 = latest["sma_50"]

        # EMA crossover detection
        bullish_cross = prev_ema_12 <= prev_ema_26 and ema_12 > ema_26
        bearish_cross = prev_ema_12 >= prev_ema_26 and ema_12 < ema_26

        # Trend strength: distance between EMAs as % of price
        ema_spread = abs(ema_12 - ema_26) / price
        confidence = min(ema_spread * 50, 1.0)  # Scale to 0-1

        if bullish_cross and price > sma_50:
            return StrategyResult(
                signal=Signal.BUY,
                confidence=max(confidence, 0.6),
                reason=f"EMA12 crossed above EMA26, price ${price:.2f} > SMA50 ${sma_50:.2f}",
                strategy_name=self.name,
            )

        if bearish_cross:
            return StrategyResult(
                signal=Signal.SELL,
                confidence=max(confidence, 0.6),
                reason=f"EMA12 crossed below EMA26",
                strategy_name=self.name,
            )

        # Sustained trend
        if ema_12 > ema_26 and price > sma_50:
            return StrategyResult(
                signal=Signal.BUY,
                confidence=confidence * 0.5,  # Lower confidence for sustained trend
                reason=f"Uptrend: EMA12 > EMA26, price above SMA50",
                strategy_name=self.name,
            )

        if ema_12 < ema_26 and price < sma_50:
            return StrategyResult(
                signal=Signal.SELL,
                confidence=confidence * 0.5,
                reason=f"Downtrend: EMA12 < EMA26, price below SMA50",
                strategy_name=self.name,
            )

        return StrategyResult(
            signal=Signal.HOLD,
            confidence=0.0,
            reason="No clear trend signal",
            strategy_name=self.name,
        )
