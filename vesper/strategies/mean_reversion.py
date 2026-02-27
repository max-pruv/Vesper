"""Mean reversion strategy — RSI + Bollinger Bands."""

from .base import Signal, Strategy, StrategyResult


class MeanReversionStrategy(Strategy):
    """
    Uses RSI and Bollinger Bands to detect overbought/oversold conditions.
    - BUY when RSI < 30 and price near lower BB
    - SELL when RSI > 70 and price near upper BB
    """

    name = "mean_reversion"

    def __init__(self, rsi_oversold: float = 30, rsi_overbought: float = 70):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def analyze(self, snapshot: dict) -> StrategyResult:
        latest = snapshot["df"].iloc[-1]

        rsi = latest["rsi"]
        price = latest["close"]
        bb_upper = latest["bb_upper"]
        bb_lower = latest["bb_lower"]
        bb_middle = latest["bb_middle"]

        # Distance from Bollinger Bands (normalized)
        bb_width = bb_upper - bb_lower
        if bb_width == 0:
            return StrategyResult(
                signal=Signal.HOLD,
                confidence=0.0,
                reason="Bollinger Bands too narrow",
                strategy_name=self.name,
            )

        price_position = (price - bb_lower) / bb_width  # 0 = at lower, 1 = at upper

        # Oversold: RSI low + price near lower BB
        if rsi < self.rsi_oversold and price_position < 0.2:
            confidence = min((self.rsi_oversold - rsi) / 30 + (0.2 - price_position), 1.0)
            return StrategyResult(
                signal=Signal.BUY,
                confidence=confidence,
                reason=f"Oversold: RSI={rsi:.1f}, price near lower BB ({price_position:.2f})",
                strategy_name=self.name,
            )

        # Overbought: RSI high + price near upper BB
        if rsi > self.rsi_overbought and price_position > 0.8:
            confidence = min((rsi - self.rsi_overbought) / 30 + (price_position - 0.8), 1.0)
            return StrategyResult(
                signal=Signal.SELL,
                confidence=confidence,
                reason=f"Overbought: RSI={rsi:.1f}, price near upper BB ({price_position:.2f})",
                strategy_name=self.name,
            )

        # Moderate signals
        if rsi < 40 and price < bb_middle:
            return StrategyResult(
                signal=Signal.BUY,
                confidence=0.3,
                reason=f"Mildly oversold: RSI={rsi:.1f}, below BB middle",
                strategy_name=self.name,
            )

        if rsi > 60 and price > bb_middle:
            return StrategyResult(
                signal=Signal.SELL,
                confidence=0.3,
                reason=f"Mildly overbought: RSI={rsi:.1f}, above BB middle",
                strategy_name=self.name,
            )

        return StrategyResult(
            signal=Signal.HOLD,
            confidence=0.0,
            reason=f"Neutral: RSI={rsi:.1f}, BB position={price_position:.2f}",
            strategy_name=self.name,
        )
