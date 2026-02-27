"""Momentum strategy — MACD + Volume confirmation."""

from .base import Signal, Strategy, StrategyResult


class MomentumStrategy(Strategy):
    """
    Uses MACD crossover confirmed by volume to detect momentum shifts.
    - BUY when MACD crosses above signal line with above-average volume
    - SELL when MACD crosses below signal line
    """

    name = "momentum"

    def analyze(self, snapshot: dict) -> StrategyResult:
        df = snapshot["df"]
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        macd = latest["macd"]
        macd_signal = latest["macd_signal"]
        macd_hist = latest["macd_hist"]
        prev_macd_hist = prev["macd_hist"]
        volume = latest["volume"]
        volume_sma = latest["volume_sma"]

        # MACD histogram crossover
        bullish_cross = prev_macd_hist <= 0 and macd_hist > 0
        bearish_cross = prev_macd_hist >= 0 and macd_hist < 0

        # Volume confirmation
        high_volume = volume > volume_sma * 1.2 if volume_sma > 0 else False

        # Momentum strength from histogram magnitude
        hist_strength = min(abs(macd_hist) / (abs(macd) + 1e-10), 1.0)

        if bullish_cross:
            confidence = 0.6 + (0.2 if high_volume else 0.0) + hist_strength * 0.2
            volume_note = " + high volume" if high_volume else ""
            return StrategyResult(
                signal=Signal.BUY,
                confidence=min(confidence, 1.0),
                reason=f"MACD bullish crossover{volume_note}",
                strategy_name=self.name,
            )

        if bearish_cross:
            confidence = 0.6 + (0.2 if high_volume else 0.0) + hist_strength * 0.2
            return StrategyResult(
                signal=Signal.SELL,
                confidence=min(confidence, 1.0),
                reason=f"MACD bearish crossover",
                strategy_name=self.name,
            )

        # Sustained momentum
        if macd_hist > 0 and macd > 0:
            return StrategyResult(
                signal=Signal.BUY,
                confidence=hist_strength * 0.4,
                reason=f"Positive MACD momentum",
                strategy_name=self.name,
            )

        if macd_hist < 0 and macd < 0:
            return StrategyResult(
                signal=Signal.SELL,
                confidence=hist_strength * 0.4,
                reason=f"Negative MACD momentum",
                strategy_name=self.name,
            )

        return StrategyResult(
            signal=Signal.HOLD,
            confidence=0.0,
            reason="No clear momentum signal",
            strategy_name=self.name,
        )
