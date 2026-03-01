"""Portfolio tracker — manages positions and P&L tracking."""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from vesper.risk import PositionLimits


@dataclass
class Position:
    symbol: str
    side: str  # "buy" or "sell"
    entry_price: float
    amount: float
    cost_usd: float
    limits: PositionLimits
    entry_time: float = field(default_factory=time.time)
    strategy_reason: str = ""
    id: str = ""
    strategy_id: str = ""
    bet_mode: str = "one_off"  # "one_off" or "continuous"
    trade_mode: str = "paper"  # "paper" or "real"
    stop_loss_pct: float = 2.0
    tp_min_pct: float = 1.5
    tp_max_pct: float = 5.0
    trailing_stop_pct: float = 0.0
    highest_price_seen: float = 0.0

    def __post_init__(self):
        if not self.id:
            self.id = f"{self.symbol}-{int(self.entry_time)}"

    def unrealized_pnl(self, current_price: float) -> float:
        if self.side == "buy":
            return (current_price - self.entry_price) * self.amount
        return (self.entry_price - current_price) * self.amount

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.side == "buy":
            return ((current_price - self.entry_price) / self.entry_price) * 100
        return ((self.entry_price - current_price) / self.entry_price) * 100


@dataclass
class TradeRecord:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    amount: float
    pnl_usd: float
    pnl_pct: float
    entry_time: float
    exit_time: float
    reason: str
    strategy_reason: str
    trade_mode: str = "paper"
    cost_usd: float = 0.0
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    total_fees: float = 0.0
    net_pnl_usd: float = 0.0
    net_pnl_pct: float = 0.0


class Portfolio:
    """Track open positions, closed trades, and overall P&L."""

    def __init__(self, initial_balance: float, data_dir: str = "data",
                 filename: str = "portfolio.json"):
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.positions: dict[str, Position] = {}
        self.trade_history: list[TradeRecord] = []
        self.data_dir = data_dir
        self.filename = filename
        os.makedirs(data_dir, exist_ok=True)
        self._load_state()

    def open_position(self, position: Position) -> bool:
        """Open a new position. Returns False if insufficient funds."""
        if position.cost_usd > self.cash:
            return False

        self.cash -= position.cost_usd
        self.positions[position.id] = position
        self._save_state()
        return True

    def close_position(
        self, position_id: str, exit_price: float, reason: str
    ) -> TradeRecord | None:
        """Close a position and record the trade."""
        pos = self.positions.get(position_id)
        if not pos:
            return None

        pnl_usd = pos.unrealized_pnl(exit_price)
        pnl_pct = pos.unrealized_pnl_pct(exit_price)

        # Calculate fees (0.6% taker fee per side for Coinbase/exchanges)
        fee_rate = 0.006
        # Check raw position data for stored entry fee
        raw = self._load_raw()
        raw_pos = raw.get("positions", {}).get(position_id, {})
        entry_fee = raw_pos.get("est_fee", round(pos.cost_usd * fee_rate, 2))
        exit_value = exit_price * pos.amount
        exit_fee = round(exit_value * fee_rate, 2)
        total_fees = round(entry_fee + exit_fee, 2)
        net_pnl_usd = round(pnl_usd - total_fees, 2)
        net_pnl_pct = round((net_pnl_usd / pos.cost_usd * 100) if pos.cost_usd > 0 else 0, 2)

        # Return capital + P&L
        self.cash += pos.cost_usd + pnl_usd

        record = TradeRecord(
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            amount=pos.amount,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            entry_time=pos.entry_time,
            exit_time=time.time(),
            reason=reason,
            strategy_reason=pos.strategy_reason,
            trade_mode=pos.trade_mode,
            cost_usd=pos.cost_usd,
            entry_fee=entry_fee,
            exit_fee=exit_fee,
            total_fees=total_fees,
            net_pnl_usd=net_pnl_usd,
            net_pnl_pct=net_pnl_pct,
        )
        self.trade_history.append(record)
        del self.positions[position_id]
        self._save_state()
        return record

    def total_value(self, prices: dict[str, float]) -> float:
        """Total portfolio value = cash + open positions value."""
        positions_value = sum(
            pos.amount * prices.get(pos.symbol, pos.entry_price)
            for pos in self.positions.values()
        )
        return self.cash + positions_value

    def get_position_for_symbol(self, symbol: str) -> Position | None:
        """Get open position for a symbol, if any."""
        for pos in self.positions.values():
            if pos.symbol == symbol:
                return pos
        return None

    def summary(self, prices: dict[str, float]) -> dict:
        """Portfolio summary stats."""
        total = self.total_value(prices)
        total_pnl = total - self.initial_balance
        total_pnl_pct = (total_pnl / self.initial_balance) * 100

        wins = [t for t in self.trade_history if t.pnl_usd > 0]
        losses = [t for t in self.trade_history if t.pnl_usd <= 0]
        total_trades = len(self.trade_history)
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0

        return {
            "total_value": total,
            "cash": self.cash,
            "total_pnl_usd": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "open_positions": len(self.positions),
            "total_trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
        }

    def _save_position_analysis(self, position_id: str, analysis: dict):
        """Store full analysis data for a position (for 'Learn More' feature)."""
        raw = self._load_raw()
        analyses = raw.get("position_analyses", {})
        analyses[position_id] = analysis
        # Prune old analyses (keep last 50)
        if len(analyses) > 50:
            keys = sorted(analyses.keys())
            for k in keys[:len(analyses) - 50]:
                del analyses[k]
        raw["position_analyses"] = analyses
        path = os.path.join(self.data_dir, self.filename)
        with open(path, "w") as f:
            json.dump(raw, f, indent=2)

    def _load_raw(self) -> dict:
        """Load the raw portfolio JSON (includes autopilot config and other metadata)."""
        path = os.path.join(self.data_dir, self.filename)
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return json.load(f)

    def _save_state(self):
        # Preserve any extra keys (e.g., autopilot config) from the existing file
        existing = self._load_raw()

        state = {
            "cash": self.cash,
            "initial_balance": self.initial_balance,
            "positions": {
                pid: {
                    "symbol": p.symbol,
                    "side": p.side,
                    "entry_price": p.entry_price,
                    "amount": p.amount,
                    "cost_usd": p.cost_usd,
                    "entry_time": p.entry_time,
                    "strategy_reason": p.strategy_reason,
                    "id": p.id,
                    "strategy_id": p.strategy_id,
                    "bet_mode": p.bet_mode,
                    "trade_mode": p.trade_mode,
                    "stop_loss_pct": p.stop_loss_pct,
                    "tp_min_pct": p.tp_min_pct,
                    "tp_max_pct": p.tp_max_pct,
                    "trailing_stop_pct": p.trailing_stop_pct,
                    "highest_price_seen": p.highest_price_seen,
                    "limits": asdict(p.limits),
                }
                for pid, p in self.positions.items()
            },
            "trade_history": [asdict(t) for t in self.trade_history],
        }
        # Carry over extra keys like "autopilot"
        for key in existing:
            if key not in state:
                state[key] = existing[key]

        path = os.path.join(self.data_dir, self.filename)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        path = os.path.join(self.data_dir, self.filename)
        if not os.path.exists(path):
            return
        with open(path) as f:
            state = json.load(f)

        self.cash = state["cash"]
        self.initial_balance = state["initial_balance"]

        self.positions = {}
        for pid, p in state.get("positions", {}).items():
            limits_data = p["limits"]
            # Ensure backward compatibility for older positions without trailing fields
            limits_data.setdefault("trailing_stop_pct", 0.0)
            limits_data.setdefault("highest_price_seen", 0.0)
            limits = PositionLimits(**limits_data)
            self.positions[pid] = Position(
                symbol=p["symbol"],
                side=p["side"],
                entry_price=p["entry_price"],
                amount=p["amount"],
                cost_usd=p["cost_usd"],
                limits=limits,
                entry_time=p["entry_time"],
                strategy_reason=p.get("strategy_reason", ""),
                id=p["id"],
                strategy_id=p.get("strategy_id", ""),
                bet_mode=p.get("bet_mode", "one_off"),
                trade_mode=p.get("trade_mode", "paper"),
                stop_loss_pct=p.get("stop_loss_pct", 2.0),
                tp_min_pct=p.get("tp_min_pct", 1.5),
                tp_max_pct=p.get("tp_max_pct", 5.0),
                trailing_stop_pct=p.get("trailing_stop_pct", 0.0),
                highest_price_seen=p.get("highest_price_seen", 0.0),
            )

        self.trade_history = [TradeRecord(**t) for t in state.get("trade_history", [])]
