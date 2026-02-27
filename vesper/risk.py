"""Risk management — stop-loss, take-profit, position sizing."""

from dataclasses import dataclass
from config.settings import RiskConfig


@dataclass
class PositionLimits:
    stop_loss_price: float
    take_profit_min_price: float
    take_profit_max_price: float
    position_size_usd: float
    position_size_asset: float
    trailing_stop_pct: float = 0.0       # 0 = disabled, >0 = active
    highest_price_seen: float = 0.0      # updated every cycle


class RiskManager:
    """Manages risk per trade: position sizing, stop-loss, take-profit."""

    def __init__(self, config: RiskConfig):
        self.config = config

    def calculate_position(
        self,
        entry_price: float,
        portfolio_value: float,
        atr: float,
        confidence: float,
        side: str = "buy",
    ) -> PositionLimits:
        """
        Calculate position size and price limits for a trade.

        Uses ATR for dynamic stop-loss when available, falls back to
        fixed percentage otherwise. Position size scales with confidence.
        """
        # Dynamic stop-loss based on ATR (2x ATR) or fixed %
        if atr > 0:
            sl_distance = atr * 2
            sl_pct = (sl_distance / entry_price) * 100
            # Clamp between 1% and configured max
            sl_pct = max(1.0, min(sl_pct, self.config.stop_loss_pct * 1.5))
        else:
            sl_pct = self.config.stop_loss_pct

        # Take-profit targets
        tp_min_pct = self.config.take_profit_min_pct
        tp_max_pct = self.config.take_profit_max_pct

        if side == "buy":
            stop_loss_price = entry_price * (1 - sl_pct / 100)
            take_profit_min = entry_price * (1 + tp_min_pct / 100)
            take_profit_max = entry_price * (1 + tp_max_pct / 100)
        else:
            stop_loss_price = entry_price * (1 + sl_pct / 100)
            take_profit_min = entry_price * (1 - tp_min_pct / 100)
            take_profit_max = entry_price * (1 - tp_max_pct / 100)

        # Position sizing: base % of portfolio, scaled by confidence
        base_pct = self.config.max_position_pct
        scaled_pct = base_pct * confidence  # Higher confidence = larger position
        scaled_pct = max(5.0, min(scaled_pct, base_pct))  # At least 5%, at most max

        position_size_usd = portfolio_value * (scaled_pct / 100)
        position_size_asset = position_size_usd / entry_price

        return PositionLimits(
            stop_loss_price=stop_loss_price,
            take_profit_min_price=take_profit_min,
            take_profit_max_price=take_profit_max,
            position_size_usd=position_size_usd,
            position_size_asset=position_size_asset,
        )

    def should_close_position(
        self,
        current_price: float,
        entry_price: float,
        limits: PositionLimits,
        side: str = "buy",
    ) -> tuple[bool, str, float]:
        """Check if a position should be closed based on risk limits.

        Returns (should_close, reason, updated_highest_price_seen).
        The 3rd value tracks the peak price for trailing stop positions.
        """
        trailing_pct = limits.trailing_stop_pct
        highest = limits.highest_price_seen

        if side == "buy":
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            new_highest = max(highest, current_price) if trailing_pct > 0 else 0.0

            if trailing_pct > 0:
                # Trailing stop mode: dynamic SL follows price up, no fixed TP max
                trailing_sl = new_highest * (1 - trailing_pct / 100)
                effective_sl = max(limits.stop_loss_price, trailing_sl)

                if current_price <= effective_sl:
                    if trailing_sl > limits.stop_loss_price:
                        return (True,
                                f"TRAILING STOP hit at ${current_price:.2f} "
                                f"(peak ${new_highest:.2f}, {pnl_pct:+.2f}%)",
                                new_highest)
                    else:
                        return (True,
                                f"STOP-LOSS hit at ${current_price:.2f} ({pnl_pct:+.2f}%)",
                                new_highest)
                return False, "", new_highest
            else:
                # Static SL/TP mode (original behavior)
                if current_price <= limits.stop_loss_price:
                    return True, f"STOP-LOSS hit at ${current_price:.2f} ({pnl_pct:+.2f}%)", 0.0

                if current_price >= limits.take_profit_max_price:
                    return True, f"MAX TAKE-PROFIT hit at ${current_price:.2f} ({pnl_pct:+.2f}%)", 0.0

        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100

            if current_price >= limits.stop_loss_price:
                return True, f"STOP-LOSS hit at ${current_price:.2f} ({pnl_pct:+.2f}%)", 0.0

            if current_price <= limits.take_profit_max_price:
                return True, f"MAX TAKE-PROFIT hit at ${current_price:.2f} ({pnl_pct:+.2f}%)", 0.0

        return False, "", new_highest if trailing_pct > 0 else 0.0

    def can_open_position(self, active_positions: int) -> bool:
        """Check if we can open a new position."""
        return active_positions < self.config.max_concurrent_positions
