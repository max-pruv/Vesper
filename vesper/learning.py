"""Vesper Learning Engine — adaptive strategy optimization based on trade history.

Analyzes closed trades to build a performance profile per strategy, then adjusts:
  1. Entry score thresholds (tighten after losses, loosen after wins)
  2. Stop-loss / take-profit ratios (learn optimal exit distances)
  3. Factor weights (boost factors that predict winning trades)
  4. Position sizing confidence multiplier

All learning parameters are stored in portfolio JSON under 'learning_state'
and applied transparently by the main trading loop.

The engine is conservative: adjustments are small (max ±10% per cycle)
and decay toward defaults if there's not enough data.
"""

import logging
import math
import time

logger = logging.getLogger("vesper.learning")

# Minimum trades needed before learning kicks in
MIN_TRADES_FOR_LEARNING = 8
# Rolling window of trades to analyze (most recent N)
ROLLING_WINDOW = 50
# Maximum adjustment from default thresholds
MAX_THRESHOLD_SHIFT = 0.15  # ±15% absolute shift on score thresholds
MAX_SL_TP_SHIFT = 0.30  # ±30% relative shift on SL/TP percentages
# Decay rate: how fast adjustments shrink when performance is neutral
DECAY_RATE = 0.05


def compute_learning_state(trade_history: list[dict], existing_state: dict | None = None) -> dict:
    """Analyze trade history and produce adaptive learning parameters.

    Args:
        trade_history: List of closed trade dicts from portfolio JSON.
        existing_state: Previous learning state (for momentum/smoothing).

    Returns:
        Dict with learning parameters to store in portfolio JSON.
    """
    if not existing_state:
        existing_state = {}

    now = time.time()

    # Filter to recent trades only
    trades = trade_history[-ROLLING_WINDOW:] if len(trade_history) > ROLLING_WINDOW else trade_history

    if len(trades) < MIN_TRADES_FOR_LEARNING:
        return {
            "updated_at": now,
            "total_trades_analyzed": len(trades),
            "status": "insufficient_data",
            "adjustments": {},
            "strategy_profiles": {},
        }

    # Group trades by strategy
    strategy_trades: dict[str, list[dict]] = {}
    for t in trades:
        sid = t.get("strategy_id", t.get("strategy_reason", "").split(":")[0].strip().lower() or "unknown")
        # Normalize strategy IDs
        if "altcoin" in sid.lower() or "hunter" in sid.lower():
            sid = "altcoin_hunter"
        elif "prediction" in sid.lower():
            sid = "predictions"
        elif "autopilot" in sid.lower():
            sid = "autopilot"
        strategy_trades.setdefault(sid, []).append(t)

    # Build per-strategy profiles
    profiles = {}
    adjustments = {}

    for sid, strades in strategy_trades.items():
        profile = _analyze_strategy(sid, strades)
        profiles[sid] = profile

        if profile["trade_count"] >= MIN_TRADES_FOR_LEARNING:
            adj = _compute_adjustments(sid, profile, existing_state.get("adjustments", {}).get(sid, {}))
            adjustments[sid] = adj

    # Global profile (all strategies combined)
    global_profile = _analyze_strategy("global", trades)
    profiles["global"] = global_profile

    return {
        "updated_at": now,
        "total_trades_analyzed": len(trades),
        "status": "active",
        "strategy_profiles": profiles,
        "adjustments": adjustments,
    }


def _analyze_strategy(strategy_id: str, trades: list[dict]) -> dict:
    """Compute performance metrics for a set of trades."""
    if not trades:
        return {"trade_count": 0}

    wins = [t for t in trades if (t.get("pnl_usd", 0) or t.get("net_pnl_usd", 0)) > 0]
    losses = [t for t in trades if (t.get("pnl_usd", 0) or t.get("net_pnl_usd", 0)) <= 0]

    win_rate = len(wins) / len(trades) if trades else 0

    # Average P&L
    pnl_values = [t.get("net_pnl_usd", t.get("pnl_usd", 0)) for t in trades]
    avg_pnl = sum(pnl_values) / len(pnl_values) if pnl_values else 0

    # Average win and loss sizes
    win_pnls = [t.get("net_pnl_pct", t.get("pnl_pct", 0)) for t in wins]
    loss_pnls = [abs(t.get("net_pnl_pct", t.get("pnl_pct", 0))) for t in losses]
    avg_win_pct = sum(win_pnls) / len(win_pnls) if win_pnls else 0
    avg_loss_pct = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0

    # Profit factor
    gross_profit = sum(max(0, p) for p in pnl_values)
    gross_loss = abs(sum(min(0, p) for p in pnl_values))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (10.0 if gross_profit > 0 else 0)

    # Win/loss streak analysis
    current_streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    streak_type = None
    for t in trades:
        pnl = t.get("pnl_usd", 0)
        if pnl > 0:
            if streak_type == "win":
                current_streak += 1
            else:
                current_streak = 1
                streak_type = "win"
            max_win_streak = max(max_win_streak, current_streak)
        else:
            if streak_type == "loss":
                current_streak += 1
            else:
                current_streak = 1
                streak_type = "loss"
            max_loss_streak = max(max_loss_streak, current_streak)

    # Trend: are recent trades better or worse than older ones?
    if len(trades) >= 6:
        recent = trades[-len(trades)//3:]
        older = trades[:len(trades)//3]
        recent_wr = sum(1 for t in recent if t.get("pnl_usd", 0) > 0) / len(recent)
        older_wr = sum(1 for t in older if t.get("pnl_usd", 0) > 0) / len(older)
        trend = "improving" if recent_wr > older_wr + 0.05 else (
            "declining" if recent_wr < older_wr - 0.05 else "stable"
        )
        trend_delta = round(recent_wr - older_wr, 3)
    else:
        trend = "insufficient_data"
        trend_delta = 0

    # Average hold duration
    durations = []
    for t in trades:
        entry_t = t.get("entry_time", 0)
        exit_t = t.get("exit_time", 0)
        if entry_t and exit_t:
            durations.append(exit_t - entry_t)
    avg_duration_h = (sum(durations) / len(durations) / 3600) if durations else 0

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        reason = t.get("reason", "unknown")
        # Normalize
        if "STOP-LOSS" in reason.upper() or "SL" in reason.upper():
            key = "stop_loss"
        elif "TAKE-PROFIT" in reason.upper() or "TP" in reason.upper() or "MAX TAKE" in reason.upper():
            key = "take_profit"
        elif "TRAILING" in reason.upper():
            key = "trailing_stop"
        elif "AI EXIT" in reason.upper() or "AI" in reason.upper():
            key = "ai_exit"
        elif "REBALANCE" in reason.upper() or "WEAKENED" in reason.upper():
            key = "rebalance"
        elif "PREDICTION" in reason.upper():
            key = "prediction_exit"
        else:
            key = "other"
        exit_reasons[key] = exit_reasons.get(key, 0) + 1

    return {
        "trade_count": len(trades),
        "win_rate": round(win_rate, 4),
        "avg_pnl_usd": round(avg_pnl, 2),
        "avg_win_pct": round(avg_win_pct, 2),
        "avg_loss_pct": round(avg_loss_pct, 2),
        "profit_factor": round(profit_factor, 2),
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "trend": trend,
        "trend_delta": trend_delta,
        "avg_duration_hours": round(avg_duration_h, 1),
        "exit_reasons": exit_reasons,
    }


def _compute_adjustments(strategy_id: str, profile: dict, prev_adj: dict) -> dict:
    """Compute parameter adjustments based on strategy performance profile.

    The learning is deliberately conservative:
    - Small shifts per update cycle
    - Decays toward zero when performance is neutral
    - Hard caps on maximum adjustment
    """
    win_rate = profile.get("win_rate", 0.5)
    profit_factor = profile.get("profit_factor", 1.0)
    trend = profile.get("trend", "stable")
    avg_win = profile.get("avg_win_pct", 2.0)
    avg_loss = profile.get("avg_loss_pct", 2.0)
    exit_reasons = profile.get("exit_reasons", {})
    trade_count = profile.get("trade_count", 0)

    # Confidence in our learning signal: more trades = higher confidence
    confidence = min(1.0, trade_count / 30)  # Full confidence at 30+ trades

    # --- 1. Entry Threshold Adjustment ---
    # If win rate is low, raise the entry threshold (be pickier)
    # If win rate is high, lower it slightly (capture more opportunities)
    prev_threshold_adj = prev_adj.get("entry_threshold_adj", 0)
    if win_rate < 0.35:
        # Poor performance: significantly raise threshold
        target_adj = 0.08 * confidence
    elif win_rate < 0.45:
        # Below average: raise threshold
        target_adj = 0.04 * confidence
    elif win_rate > 0.60:
        # Good performance: lower threshold slightly
        target_adj = -0.03 * confidence
    elif win_rate > 0.55:
        # Slightly above average: tiny adjustment
        target_adj = -0.01 * confidence
    else:
        # Neutral: decay toward zero
        target_adj = prev_threshold_adj * (1 - DECAY_RATE)

    # Smooth: move 30% toward target each cycle
    entry_threshold_adj = prev_threshold_adj + 0.3 * (target_adj - prev_threshold_adj)
    entry_threshold_adj = max(-MAX_THRESHOLD_SHIFT, min(MAX_THRESHOLD_SHIFT, entry_threshold_adj))

    # If performance is declining, add urgency
    if trend == "declining":
        entry_threshold_adj = max(entry_threshold_adj, 0.02 * confidence)

    # --- 2. Stop-Loss Adjustment ---
    # If most exits are SL hits and avg loss is large, tighten SL
    # If SL hits too often on trades that would have recovered, loosen
    sl_exits = exit_reasons.get("stop_loss", 0)
    total_exits = sum(exit_reasons.values()) or 1
    sl_ratio = sl_exits / total_exits

    prev_sl_adj = prev_adj.get("stop_loss_adj_pct", 0)
    if sl_ratio > 0.60 and win_rate < 0.45:
        # Too many SL hits AND losing: tighten to cut losses faster
        target_sl_adj = -0.15 * confidence  # Tighter SL
    elif sl_ratio > 0.50 and win_rate > 0.50:
        # Many SL hits but overall profitable: SL might be too tight
        target_sl_adj = 0.10 * confidence  # Loosen SL
    elif sl_ratio < 0.20 and avg_loss > avg_win:
        # Not hitting SL but losses are big: SL not tight enough
        target_sl_adj = -0.10 * confidence
    else:
        target_sl_adj = prev_sl_adj * (1 - DECAY_RATE)

    stop_loss_adj_pct = prev_sl_adj + 0.3 * (target_sl_adj - prev_sl_adj)
    stop_loss_adj_pct = max(-MAX_SL_TP_SHIFT, min(MAX_SL_TP_SHIFT, stop_loss_adj_pct))

    # --- 3. Take-Profit Adjustment ---
    # If avg win is much smaller than avg loss, TP might be too tight
    # If avg win is much larger, TP is well calibrated or could be tighter
    prev_tp_adj = prev_adj.get("take_profit_adj_pct", 0)
    if avg_win > 0 and avg_loss > 0:
        rr_ratio = avg_win / avg_loss  # Risk-reward ratio
        if rr_ratio < 0.8:
            # Winning less than losing: widen TP to let winners run
            target_tp_adj = 0.15 * confidence
        elif rr_ratio > 2.0:
            # Great R:R but maybe missing smaller wins: tighten TP slightly
            target_tp_adj = -0.05 * confidence
        else:
            target_tp_adj = prev_tp_adj * (1 - DECAY_RATE)
    else:
        target_tp_adj = prev_tp_adj * (1 - DECAY_RATE)

    take_profit_adj_pct = prev_tp_adj + 0.3 * (target_tp_adj - prev_tp_adj)
    take_profit_adj_pct = max(-MAX_SL_TP_SHIFT, min(MAX_SL_TP_SHIFT, take_profit_adj_pct))

    # --- 4. Position Size Confidence Multiplier ---
    # Scale position sizes based on recent performance
    prev_size_mult = prev_adj.get("position_size_multiplier", 1.0)
    if profit_factor > 1.5 and win_rate > 0.50:
        target_mult = 1.0 + 0.15 * confidence  # Up to +15% size
    elif profit_factor < 0.8 or win_rate < 0.35:
        target_mult = 1.0 - 0.20 * confidence  # Down to -20% size
    else:
        target_mult = 1.0 + (prev_size_mult - 1.0) * (1 - DECAY_RATE)

    position_size_multiplier = prev_size_mult + 0.2 * (target_mult - prev_size_mult)
    position_size_multiplier = max(0.5, min(1.3, position_size_multiplier))

    # --- 5. Rebalance Threshold Adjustment (altcoin_hunter) ---
    # If rebalance exits are mostly losers, lower the rebalance threshold
    rebalance_exits = exit_reasons.get("rebalance", 0)
    prev_rebalance_adj = prev_adj.get("rebalance_threshold_adj", 0)
    if rebalance_exits > 3 and win_rate < 0.40:
        target_rebalance = -0.05 * confidence  # Rebalance sooner
    elif rebalance_exits > 3 and win_rate > 0.55:
        target_rebalance = 0.03 * confidence  # More patience
    else:
        target_rebalance = prev_rebalance_adj * (1 - DECAY_RATE)

    rebalance_threshold_adj = prev_rebalance_adj + 0.3 * (target_rebalance - prev_rebalance_adj)
    rebalance_threshold_adj = max(-0.10, min(0.10, rebalance_threshold_adj))

    return {
        "entry_threshold_adj": round(entry_threshold_adj, 4),
        "stop_loss_adj_pct": round(stop_loss_adj_pct, 4),
        "take_profit_adj_pct": round(take_profit_adj_pct, 4),
        "position_size_multiplier": round(position_size_multiplier, 4),
        "rebalance_threshold_adj": round(rebalance_threshold_adj, 4),
        "learning_confidence": round(confidence, 3),
    }


def apply_entry_threshold(base_threshold: float, learning_state: dict, strategy_id: str) -> float:
    """Apply learned adjustment to an entry score threshold.

    Returns adjusted threshold (higher = pickier, lower = more permissive).
    """
    adj = _get_adjustment(learning_state, strategy_id)
    if not adj:
        return base_threshold
    shift = adj.get("entry_threshold_adj", 0)
    return max(0.20, min(0.80, base_threshold + shift))


def apply_stop_loss(base_sl_pct: float, learning_state: dict, strategy_id: str) -> float:
    """Apply learned adjustment to stop-loss percentage."""
    adj = _get_adjustment(learning_state, strategy_id)
    if not adj:
        return base_sl_pct
    shift = adj.get("stop_loss_adj_pct", 0)
    return max(0.5, base_sl_pct * (1 + shift))


def apply_take_profit(base_tp_pct: float, learning_state: dict, strategy_id: str) -> float:
    """Apply learned adjustment to take-profit percentage."""
    adj = _get_adjustment(learning_state, strategy_id)
    if not adj:
        return base_tp_pct
    shift = adj.get("take_profit_adj_pct", 0)
    return max(0.5, base_tp_pct * (1 + shift))


def apply_position_size(base_size_usd: float, learning_state: dict, strategy_id: str) -> float:
    """Apply learned position size multiplier."""
    adj = _get_adjustment(learning_state, strategy_id)
    if not adj:
        return base_size_usd
    mult = adj.get("position_size_multiplier", 1.0)
    return base_size_usd * mult


def apply_rebalance_threshold(base_threshold: float, learning_state: dict) -> float:
    """Apply learned adjustment to the altcoin hunter rebalance exit threshold."""
    adj = _get_adjustment(learning_state, "altcoin_hunter")
    if not adj:
        return base_threshold
    shift = adj.get("rebalance_threshold_adj", 0)
    return max(0.20, min(0.60, base_threshold + shift))


def _get_adjustment(learning_state: dict, strategy_id: str) -> dict:
    """Get adjustments for a strategy, falling back to global if not available."""
    if not learning_state or learning_state.get("status") != "active":
        return {}
    adjustments = learning_state.get("adjustments", {})
    return adjustments.get(strategy_id, adjustments.get("global", {}))
