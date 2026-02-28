"""Vesper — Main orchestrator and scheduler for multi-user trading."""

import sys
import os
import signal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading

from apscheduler.schedulers.blocking import BlockingScheduler
import uvicorn

from config.settings import ExchangeConfig, RiskConfig, TICKER_SYMBOLS, STOCK_SYMBOLS, ALTCOIN_UNIVERSE, is_stock_symbol
from vesper.exchange import create_exchange, AlpacaExchange
from vesper.market_data import (
    get_market_snapshot, get_multi_tf_snapshot,
    get_order_book_pressure, fetch_fear_greed,
    enrich_with_intelligence, get_stock_snapshot,
)
from vesper.strategies import (
    Signal, EnsembleStrategy, EnhancedEnsemble,
    TrendFollowingStrategy, MeanReversionStrategy, MomentumStrategy,
)
from vesper.strategies.altcoin_hunter import AltcoinHunterStrategy
from vesper.risk import RiskManager, PositionLimits
from vesper.portfolio import Portfolio, Position
from vesper.logger import setup_logger, console


# Maps strategy_id → (StrategyClass, timeframe)
# Each strategy uses a genuinely different analysis engine + timeframe
STRATEGY_MAP = {
    "scalper":        {"strategy": MomentumStrategy,       "timeframe": "15m"},
    "trend_rider":    {"strategy": TrendFollowingStrategy,  "timeframe": "4h"},
    "mean_revert":    {"strategy": MeanReversionStrategy,   "timeframe": "1h"},
    "smart_auto":     {"strategy": EnhancedEnsemble,        "timeframe": "multi"},
    "trend_scanner":  {"strategy": EnhancedEnsemble,        "timeframe": "multi"},
    "autopilot":      {"strategy": EnhancedEnsemble,        "timeframe": "multi"},
    "altcoin_hunter": {"strategy": AltcoinHunterStrategy,   "timeframe": "multi"},
    "set_and_forget": {"strategy": None,                    "timeframe": None},
}


class UserBot:
    """Trading bot instance for a single user."""

    def __init__(self, user_id: int, email: str, exchange_cfg: ExchangeConfig,
                 risk_cfg: RiskConfig, symbols: list[str], mode: str,
                 paper_balance: float, logger, alpaca_client=None):
        self.user_id = user_id
        self.email = email
        self.mode = mode
        self.symbols = symbols
        self.logger = logger
        self.risk_manager = RiskManager(risk_cfg)
        # Strategy instances are resolved per-position from STRATEGY_MAP
        self._strategy_cache: dict[str, object] = {}

        data_dir = os.environ.get("VESPER_DATA_DIR", "data")
        self.portfolio = Portfolio(
            initial_balance=paper_balance,
            data_dir=data_dir,
            filename=f"portfolio_{user_id}.json",
        )

        self.exchange = create_exchange(exchange_cfg)
        self.alpaca = alpaca_client  # None if user hasn't connected Alpaca

    def run_cycle(self):
        """Run one trading cycle for this user."""
        self.logger.info(f"[User:{self.email}] Starting cycle")
        prices = {}

        # Combine configured symbols + symbols with open positions
        open_symbols = {pos.symbol for pos in self.portfolio.positions.values()}
        all_symbols = list(dict.fromkeys(self.symbols + list(open_symbols)))

        for symbol in all_symbols:
            try:
                self._process_symbol(symbol, prices)
            except Exception as e:
                self.logger.error(f"[User:{self.email}] Error on {symbol}: {e}")

        # Trend Scanner: scan broader market for opportunities
        self._run_trend_scanner(prices)

        # Autopilot: AI fully manages allocated funds
        self._run_autopilot(prices)

        # Altcoin Hunter: autonomous altcoin trend detection + self-investing
        self._run_altcoin_hunter(prices)

        summary = self.portfolio.summary(prices)
        self.logger.info(
            f"[User:{self.email}] Portfolio: ${summary['total_value']:,.2f} | "
            f"P&L: ${summary['total_pnl_usd']:+,.2f} ({summary['total_pnl_pct']:+.2f}%)"
        )

    def _get_strategy(self, strategy_id: str):
        """Get or create a strategy instance for the given strategy_id."""
        if strategy_id in self._strategy_cache:
            return self._strategy_cache[strategy_id]
        config = STRATEGY_MAP.get(strategy_id)
        if not config or config["strategy"] is None:
            return None
        instance = config["strategy"]()
        self._strategy_cache[strategy_id] = instance
        return instance

    def _get_snapshot(self, symbol: str, strategy_id: str) -> dict:
        """Fetch the right market snapshot for a strategy's timeframe."""
        # Stock symbols use Alpaca data
        if is_stock_symbol(symbol) and self.alpaca:
            return get_stock_snapshot(self.alpaca, symbol)

        config = STRATEGY_MAP.get(strategy_id, {})
        timeframe = config.get("timeframe", "1h")

        if timeframe == "multi":
            # Multi-timeframe: 1h + 4h with alignment
            snapshot = get_multi_tf_snapshot(self.exchange, symbol)
            # Enrich with order book + sentiment for enhanced strategies
            try:
                ob = get_order_book_pressure(self.exchange, symbol)
                snapshot["buy_pressure"] = ob["buy_pressure"]
                snapshot["spread_pct"] = ob["spread_pct"]
            except Exception:
                snapshot["buy_pressure"] = 0.5
            try:
                snapshot["fear_greed"] = fetch_fear_greed()
            except Exception:
                snapshot["fear_greed"] = 50
            # Enrich with whale tracking + composite sentiment (AI intelligence layer)
            try:
                enrich_with_intelligence(self.exchange, symbol, snapshot)
            except Exception:
                pass
        elif timeframe:
            snapshot = get_market_snapshot(self.exchange, symbol, timeframe=timeframe)
        else:
            snapshot = get_market_snapshot(self.exchange, symbol)

        return snapshot

    def _snapshot_indicators(self, snapshot: dict) -> dict:
        """Extract key indicators from a snapshot for the decision log."""
        return {
            "price": round(snapshot.get("price", 0), 2),
            "rsi": round(snapshot.get("rsi", 0), 1),
            "macd_hist": round(snapshot.get("macd_hist", 0), 4),
            "ema_trend": "bullish" if snapshot.get("ema_12", 0) > snapshot.get("ema_26", 0) else "bearish",
            "adx": round(snapshot.get("adx", 0), 1),
            "atr": round(snapshot.get("atr", 0), 2),
            "bb_position": round(
                ((snapshot.get("price", 0) - snapshot.get("bb_lower", 0))
                 / max(snapshot.get("bb_upper", 1) - snapshot.get("bb_lower", 1), 0.01)) * 100, 1
            ),
            "whale_score": round(snapshot.get("whale_score", 0), 2),
            "sentiment": round(snapshot.get("sentiment_score", 0), 2),
            "fear_greed": snapshot.get("fear_greed", None),
            "tf_alignment": snapshot.get("tf_alignment", None),
        }

    def _process_symbol(self, symbol: str, prices: dict):
        # Find existing position for this symbol (if any)
        existing_pos = self.portfolio.get_position_for_symbol(symbol)
        strategy_id = existing_pos.strategy_id if existing_pos else "smart_auto"

        # Set & Forget: only check SL/TP, never auto-open
        if strategy_id == "set_and_forget":
            if not existing_pos:
                return
            snapshot = get_market_snapshot(self.exchange, symbol)
            prices[symbol] = snapshot["price"]
            should_close, reason, new_highest = self.risk_manager.should_close_position(
                current_price=snapshot["price"],
                entry_price=existing_pos.entry_price,
                limits=existing_pos.limits,
                side=existing_pos.side,
            )
            if should_close:
                self._save_decision({
                    "action": "EXIT", "symbol": symbol, "strategy_id": strategy_id,
                    "source": "cycle", "trade_mode": self.mode,
                    "reason": reason,
                    "indicators": self._snapshot_indicators(snapshot),
                })
                self._close_position(existing_pos, snapshot["price"], reason)
            elif new_highest > existing_pos.highest_price_seen:
                existing_pos.highest_price_seen = new_highest
                existing_pos.limits.highest_price_seen = new_highest
                self.portfolio._save_state()
            return

        # Fetch snapshot using the strategy's timeframe
        snapshot = self._get_snapshot(symbol, strategy_id)
        prices[symbol] = snapshot["price"]

        closed_continuous = None
        if existing_pos:
            should_close, reason, new_highest = self.risk_manager.should_close_position(
                current_price=snapshot["price"],
                entry_price=existing_pos.entry_price,
                limits=existing_pos.limits,
                side=existing_pos.side,
            )
            if should_close:
                pnl_pct = ((snapshot["price"] - existing_pos.entry_price)
                           / existing_pos.entry_price) * 100
                self._save_decision({
                    "action": "EXIT", "symbol": symbol, "strategy_id": strategy_id,
                    "source": "cycle", "trade_mode": self.mode,
                    "reason": reason,
                    "pnl_pct": round(pnl_pct, 2),
                    "indicators": self._snapshot_indicators(snapshot),
                })
                if existing_pos.bet_mode == "continuous":
                    closed_continuous = existing_pos
                self._close_position(existing_pos, snapshot["price"], reason)
                if closed_continuous is None:
                    return  # One-off: done
            else:
                # ── AI-driven exit: check if AI recommends closing ──
                strategy = self._get_strategy(strategy_id)
                if strategy:
                    result = strategy.analyze(snapshot)
                    if result.signal == Signal.SELL and result.confidence >= 0.6:
                        pnl_pct = ((snapshot["price"] - existing_pos.entry_price)
                                   / existing_pos.entry_price) * 100
                        ai_reason = (
                            f"AI EXIT ({result.confidence:.0%}): {result.reason[:100]}"
                        )
                        self.logger.info(
                            f"[User:{self.email}] AI recommends exit for "
                            f"{symbol} (P&L: {pnl_pct:+.1f}%) — {ai_reason[:80]}"
                        )
                        self._save_decision({
                            "action": "EXIT", "symbol": symbol, "strategy_id": strategy_id,
                            "source": "ai_exit", "trade_mode": self.mode,
                            "signal": "SELL", "confidence": round(result.confidence, 3),
                            "reason": result.reason,
                            "pnl_pct": round(pnl_pct, 2),
                            "indicators": self._snapshot_indicators(snapshot),
                        })
                        if existing_pos.bet_mode == "continuous":
                            closed_continuous = existing_pos
                        self._close_position(existing_pos, snapshot["price"], ai_reason)
                        if closed_continuous is None:
                            return
                    else:
                        # Update trailing stop tracking
                        if new_highest > existing_pos.highest_price_seen:
                            existing_pos.highest_price_seen = new_highest
                            existing_pos.limits.highest_price_seen = new_highest
                            self.portfolio._save_state()
                        return  # Position still open, nothing to do
                else:
                    # Update trailing stop tracking
                    if new_highest > existing_pos.highest_price_seen:
                        existing_pos.highest_price_seen = new_highest
                        existing_pos.limits.highest_price_seen = new_highest
                        self.portfolio._save_state()
                    return

        # Analyze with the correct strategy
        strategy = self._get_strategy(strategy_id)
        if not strategy:
            return
        result = strategy.analyze(snapshot)

        if result.signal == Signal.HOLD:
            self._save_decision({
                "action": "SKIP", "symbol": symbol, "strategy_id": strategy_id,
                "source": "cycle", "trade_mode": self.mode,
                "signal": "HOLD", "confidence": round(result.confidence, 3),
                "reason": result.reason,
                "indicators": self._snapshot_indicators(snapshot),
            })
            return

        if not self.risk_manager.can_open_position(len(self.portfolio.positions)):
            self._save_decision({
                "action": "SKIP", "symbol": symbol, "strategy_id": strategy_id,
                "source": "cycle", "trade_mode": self.mode,
                "signal": result.signal.name, "confidence": round(result.confidence, 3),
                "reason": f"Max positions reached. Signal was: {result.reason}",
                "indicators": self._snapshot_indicators(snapshot),
            })
            return

        if result.signal == Signal.BUY:
            self._save_decision({
                "action": "ENTER_LONG", "symbol": symbol, "strategy_id": strategy_id,
                "source": "cycle", "trade_mode": self.mode,
                "signal": "BUY", "confidence": round(result.confidence, 3),
                "reason": result.reason,
                "indicators": self._snapshot_indicators(snapshot),
            })
            if closed_continuous:
                self._reopen_continuous(closed_continuous, snapshot)
            else:
                self._open_position(symbol, snapshot, result)
        elif result.signal == Signal.SELL:
            self._save_decision({
                "action": "SIGNAL_SELL", "symbol": symbol, "strategy_id": strategy_id,
                "source": "cycle", "trade_mode": self.mode,
                "signal": "SELL", "confidence": round(result.confidence, 3),
                "reason": result.reason,
                "indicators": self._snapshot_indicators(snapshot),
            })

    def _open_position(self, symbol, snapshot, result):
        portfolio_value = self.portfolio.total_value({symbol: snapshot["price"]})
        limits = self.risk_manager.calculate_position(
            entry_price=snapshot["price"],
            portfolio_value=portfolio_value,
            atr=snapshot["atr"],
            confidence=result.confidence,
            side="buy",
        )

        position = Position(
            symbol=symbol, side="buy",
            entry_price=snapshot["price"],
            amount=limits.position_size_asset,
            cost_usd=limits.position_size_usd,
            limits=limits,
            strategy_reason=result.reason,
        )

        if self.mode == "paper":
            if self.portfolio.open_position(position):
                self.logger.info(
                    f"[User:{self.email}] PAPER BUY {symbol}: "
                    f"{position.amount:.6f} @ ${position.entry_price:,.2f}"
                )
        else:
            try:
                if is_stock_symbol(symbol) and self.alpaca:
                    from vesper.exchange import alpaca_market_buy
                    order = alpaca_market_buy(self.alpaca, symbol, limits.position_size_usd)
                else:
                    from vesper.exchange import place_market_buy
                    order = place_market_buy(self.exchange, symbol, limits.position_size_asset)
                position.entry_price = float(order.get("average", snapshot["price"]))
                position.amount = float(order.get("filled", limits.position_size_asset))
                position.cost_usd = position.entry_price * position.amount
                self.portfolio.open_position(position)
                self.logger.info(f"[User:{self.email}] LIVE BUY {symbol}: filled")
            except Exception as e:
                self.logger.error(f"[User:{self.email}] BUY failed {symbol}: {e}")

    def _close_position(self, position: Position, exit_price: float, reason: str):
        trade_mode = position.trade_mode if position.trade_mode else self.mode
        if trade_mode == "real" or self.mode == "live":
            try:
                if is_stock_symbol(position.symbol) and self.alpaca:
                    from vesper.exchange import alpaca_market_sell
                    alpaca_market_sell(self.alpaca, position.symbol, position.amount)
                else:
                    from vesper.exchange import place_market_sell
                    place_market_sell(self.exchange, position.symbol, position.amount)
            except Exception as e:
                self.logger.error(f"[User:{self.email}] SELL failed: {e}")
                return

        record = self.portfolio.close_position(position.id, exit_price, reason)
        if record:
            self.logger.info(
                f"[User:{self.email}] CLOSED {position.symbol}: "
                f"${record.pnl_usd:+,.2f} ({record.pnl_pct:+.2f}%) — {reason}"
            )

    def _reopen_continuous(self, old_pos: Position, snapshot: dict):
        """Re-open a continuous bet with the same parameters at the new price."""
        new_price = snapshot["price"]
        amount_usd = old_pos.cost_usd
        sl_pct = old_pos.stop_loss_pct
        tp_min_pct = old_pos.tp_min_pct
        tp_max_pct = old_pos.tp_max_pct

        trailing_pct = old_pos.trailing_stop_pct

        # When trailing is active, TP max is a sentinel (let it run)
        effective_tp_max = new_price * 100 if trailing_pct > 0 else new_price * (1 + tp_max_pct / 100)

        limits = PositionLimits(
            stop_loss_price=new_price * (1 - sl_pct / 100),
            take_profit_min_price=new_price * (1 + tp_min_pct / 100),
            take_profit_max_price=effective_tp_max,
            position_size_usd=amount_usd,
            position_size_asset=amount_usd / new_price,
            trailing_stop_pct=trailing_pct,
            highest_price_seen=new_price,
        )

        position = Position(
            symbol=old_pos.symbol, side="buy",
            entry_price=new_price,
            amount=limits.position_size_asset,
            cost_usd=amount_usd,
            limits=limits,
            strategy_reason=f"Continuous re-entry",
            strategy_id=old_pos.strategy_id,
            bet_mode="continuous",
            trade_mode=old_pos.trade_mode,
            stop_loss_pct=sl_pct,
            tp_min_pct=tp_min_pct,
            tp_max_pct=tp_max_pct,
            trailing_stop_pct=trailing_pct,
            highest_price_seen=new_price,
        )

        trade_mode = old_pos.trade_mode or self.mode
        if trade_mode == "real":
            try:
                from vesper.exchange import place_market_buy
                order = place_market_buy(self.exchange, old_pos.symbol, limits.position_size_asset)
                position.entry_price = float(order.get("average", new_price))
                position.amount = float(order.get("filled", limits.position_size_asset))
                position.cost_usd = position.entry_price * position.amount
            except Exception as e:
                self.logger.error(f"[User:{self.email}] Continuous re-entry BUY failed: {e}")
                return

        if self.portfolio.open_position(position):
            self.logger.info(
                f"[User:{self.email}] CONTINUOUS RE-ENTRY {old_pos.symbol}: "
                f"${amount_usd:.2f} @ ${new_price:,.2f}"
            )

    def _run_trend_scanner(self, prices: dict):
        """Scan the broader market for trend_scanner positions."""
        # Check if user has any trend_scanner continuous positions
        has_scanner = any(
            pos.strategy_id == "trend_scanner"
            for pos in self.portfolio.positions.values()
        )
        # If there's already an open trend_scanner position, skip scanning
        if has_scanner:
            return

        # Check if user has ever placed a trend_scanner bet (look in trade history)
        # Only scan if the user has actively used this strategy
        scanner_history = [
            t for t in self.portfolio.trade_history
            if "trend_scanner" in (t.strategy_reason or "").lower()
        ]
        if not scanner_history:
            return

        if not self.risk_manager.can_open_position(len(self.portfolio.positions)):
            return

        # Use the most recent trend_scanner trade for parameters
        last_scanner = scanner_history[-1]

        # Use EnhancedEnsemble with multi-TF for scanning
        scanner_strategy = self._get_strategy("trend_scanner") or EnhancedEnsemble()

        # Scan all major symbols for the strongest BUY signal
        best_signal = None
        best_snapshot = None
        best_symbol = None

        for sym in TICKER_SYMBOLS:
            try:
                if sym in prices:
                    continue
                snap = self._get_snapshot(sym, "trend_scanner")
                prices[sym] = snap["price"]
                result = scanner_strategy.analyze(snap)
                if result.signal == Signal.BUY:
                    if best_signal is None or result.confidence > best_signal.confidence:
                        best_signal = result
                        best_snapshot = snap
                        best_symbol = sym
            except Exception:
                continue

        if best_signal and best_symbol and best_signal.confidence >= 0.6:
            # Open position on the strongest signal
            # Use cost from last scanner trade as position size
            amount_usd = last_scanner.amount * last_scanner.entry_price
            if amount_usd <= 0:
                amount_usd = 50.0  # Fallback default
            new_price = best_snapshot["price"]

            from vesper.risk import PositionLimits
            sl_pct = 2.0
            tp_max_pct = 5.0
            limits = PositionLimits(
                stop_loss_price=new_price * (1 - sl_pct / 100),
                take_profit_min_price=new_price * (1 + 1.5 / 100),
                take_profit_max_price=new_price * (1 + tp_max_pct / 100),
                position_size_usd=amount_usd,
                position_size_asset=amount_usd / new_price,
            )

            position = Position(
                symbol=best_symbol, side="buy",
                entry_price=new_price,
                amount=limits.position_size_asset,
                cost_usd=amount_usd,
                limits=limits,
                strategy_reason=f"Trend Scanner: {best_signal.reason}",
                strategy_id="trend_scanner",
                bet_mode="continuous",
                trade_mode="paper",
                stop_loss_pct=sl_pct,
                tp_min_pct=1.5,
                tp_max_pct=tp_max_pct,
            )

            if self.portfolio.open_position(position):
                self.logger.info(
                    f"[User:{self.email}] TREND SCANNER: {best_symbol} "
                    f"BUY ${amount_usd:.2f} @ ${new_price:,.2f} "
                    f"(confidence: {best_signal.confidence:.0%})"
                )

    def _run_altcoin_hunter(self, prices: dict):
        """Altcoin Hunter — autonomous trend detection, self-investing, self-regulating.

        Scans 50+ altcoins, scores each on 6 trend factors, dynamically allocates
        capital to the strongest trends, and auto-rebalances by exiting weakening
        positions and entering new strong ones.
        """
        pdata = self.portfolio._load_raw()
        hunter_cfg = pdata.get("altcoin_hunter")
        if not hunter_cfg or not hunter_cfg.get("enabled"):
            return

        fund_total = hunter_cfg.get("fund_usd", 0)
        if fund_total <= 0:
            return

        max_positions = hunter_cfg.get("max_positions", 5)
        trailing_pct = hunter_cfg.get("trailing_stop_pct", 2.0)
        min_score = hunter_cfg.get("min_trend_score", 0.60)

        # Collect existing altcoin_hunter positions
        hunter_positions = [
            pos for pos in self.portfolio.positions.values()
            if pos.strategy_id == "altcoin_hunter"
        ]

        deployed = sum(pos.cost_usd for pos in hunter_positions)
        available = fund_total - deployed
        slots_open = max_positions - len(hunter_positions)

        # ── Phase 1: Get BTC snapshot for relative strength comparison ──
        btc_snapshot = None
        try:
            btc_snapshot = self._get_snapshot("BTC/USDT", "altcoin_hunter")
            prices["BTC/USDT"] = btc_snapshot["price"]
        except Exception:
            pass

        # ── Phase 2: Score all altcoins ──
        from vesper.strategies.altcoin_hunter import compute_trend_score
        scored = []
        scanned = 0

        for sym in ALTCOIN_UNIVERSE:
            if sym == "BTC/USDT":
                continue  # Skip BTC itself (we compare against it)
            try:
                snap = self._get_snapshot(sym, "altcoin_hunter")
                prices[sym] = snap["price"]
                score_data = compute_trend_score(snap, btc_snapshot)
                scanned += 1
                scored.append({
                    "symbol": sym,
                    "score": score_data["score"],
                    "signal": score_data["signal"],
                    "confidence": score_data["confidence"],
                    "factors": score_data["factors"],
                    "snapshot": snap,
                })
            except Exception:
                continue

        # Sort by score descending
        scored.sort(key=lambda x: x["score"], reverse=True)

        # ── Phase 3: Rebalance — close positions that lost momentum ──
        actions = []
        for pos in list(hunter_positions):
            sym_data = next((s for s in scored if s["symbol"] == pos.symbol), None)
            if sym_data and sym_data["score"] < 0.40:
                # Trend has weakened significantly — exit
                exit_price = prices.get(pos.symbol, pos.entry_price)
                pnl_pct = ((exit_price - pos.entry_price) / pos.entry_price) * 100
                reason = (
                    f"Altcoin Hunter rebalance: trend weakened "
                    f"(score {sym_data['score']:.2f} < 0.40, P&L {pnl_pct:+.1f}%)"
                )
                self._save_decision({
                    "action": "EXIT", "symbol": pos.symbol,
                    "strategy_id": "altcoin_hunter",
                    "source": "altcoin_hunter_rebalance",
                    "trade_mode": self.mode,
                    "reason": reason,
                    "pnl_pct": round(pnl_pct, 2),
                })
                self._close_position(pos, exit_price, reason)
                deployed -= pos.cost_usd
                available += pos.cost_usd
                slots_open += 1
                actions.append({
                    "action": "SELL", "symbol": pos.symbol,
                    "reason": "trend_weakened",
                    "score": sym_data["score"],
                    "pnl_pct": round(pnl_pct, 2),
                })

        # ── Phase 4: Open new positions on the strongest trends ──
        if slots_open <= 0 or available < 10:
            self._save_autopilot_log({
                "type": "altcoin_hunter_scan",
                "symbols_scanned": scanned,
                "status": "fully_deployed" if slots_open <= 0 else "low_funds",
                "positions": max_positions - slots_open,
                "max_positions": max_positions,
                "deployed_usd": round(deployed, 2),
                "available_usd": round(available, 2),
                "top_scores": [
                    {"symbol": s["symbol"], "score": s["score"]}
                    for s in scored[:10]
                ],
                "actions": actions,
            })
            return

        # Filter candidates: strong uptrend + not already holding
        held_symbols = {pos.symbol for pos in self.portfolio.positions.values()}
        candidates = [
            s for s in scored
            if s["score"] >= min_score
            and s["signal"] == Signal.BUY
            and s["symbol"] not in held_symbols
        ]

        to_open = candidates[:slots_open]
        if not to_open:
            self._save_autopilot_log({
                "type": "altcoin_hunter_scan",
                "symbols_scanned": scanned,
                "status": "no_opportunity",
                "positions": max_positions - slots_open,
                "max_positions": max_positions,
                "deployed_usd": round(deployed, 2),
                "available_usd": round(available, 2),
                "top_scores": [
                    {"symbol": s["symbol"], "score": s["score"]}
                    for s in scored[:10]
                ],
                "actions": actions,
            })
            return

        # Dynamic allocation: weight by score (stronger trend = more capital)
        total_score = sum(c["score"] for c in to_open)
        for candidate in to_open:
            weight = candidate["score"] / total_score if total_score > 0 else 1 / len(to_open)
            amount_usd = min(available * weight, available)
            if amount_usd < 10:
                continue

            sym = candidate["symbol"]
            snap = candidate["snapshot"]
            new_price = snap["price"]

            # Adaptive stop-loss: tighter in volatile markets
            atr = snap.get("atr", 0)
            if atr and new_price > 0:
                sl_pct = min(max((2.5 * atr / new_price) * 100, 1.5), 5.0)
            else:
                sl_pct = 2.5

            # Dynamic TP scaled by trend score
            score = candidate["score"]
            tp_min_pct = 2.0 + score * 2.0   # 2% to 4%
            tp_max_pct = 8.0 + score * 12.0  # 8% to 20%

            limits = PositionLimits(
                stop_loss_price=new_price * (1 - sl_pct / 100),
                take_profit_min_price=new_price * (1 + tp_min_pct / 100),
                take_profit_max_price=new_price * (1 + tp_max_pct / 100),
                position_size_usd=amount_usd,
                position_size_asset=amount_usd / new_price,
                trailing_stop_pct=trailing_pct,
                highest_price_seen=new_price,
            )

            position = Position(
                symbol=sym, side="buy",
                entry_price=new_price,
                amount=limits.position_size_asset,
                cost_usd=amount_usd,
                limits=limits,
                strategy_reason=(
                    f"Altcoin Hunter: score {score:.2f} | "
                    f"{', '.join(f'{k}={v:.2f}' for k, v in candidate['factors'].items())}"
                ),
                strategy_id="altcoin_hunter",
                bet_mode="continuous",
                trade_mode=self.mode if self.mode == "live" else "paper",
                stop_loss_pct=sl_pct,
                tp_min_pct=tp_min_pct,
                tp_max_pct=tp_max_pct,
                trailing_stop_pct=trailing_pct,
                highest_price_seen=new_price,
            )

            # Execute trade
            if self.mode == "live":
                try:
                    from vesper.exchange import place_market_buy
                    order = place_market_buy(self.exchange, sym, limits.position_size_asset)
                    position.entry_price = float(order.get("average", new_price))
                    position.amount = float(order.get("filled", limits.position_size_asset))
                    position.cost_usd = position.entry_price * position.amount
                except Exception as e:
                    self.logger.error(f"[User:{self.email}] Altcoin Hunter BUY failed {sym}: {e}")
                    continue

            if self.portfolio.open_position(position):
                available -= amount_usd
                deployed += amount_usd
                slots_open -= 1
                actions.append({
                    "action": "BUY", "symbol": sym,
                    "amount_usd": round(amount_usd, 2),
                    "price": round(new_price, 2),
                    "score": score,
                    "confidence": candidate["confidence"],
                    "factors": candidate["factors"],
                })
                self._save_decision({
                    "action": "ENTER_LONG", "symbol": sym,
                    "strategy_id": "altcoin_hunter",
                    "source": "altcoin_hunter",
                    "trade_mode": self.mode if self.mode == "live" else "paper",
                    "signal": "BUY",
                    "confidence": round(candidate["confidence"], 3),
                    "reason": position.strategy_reason,
                    "amount_usd": round(amount_usd, 2),
                    "indicators": self._snapshot_indicators(snap),
                })
                self.logger.info(
                    f"[User:{self.email}] ALTCOIN HUNTER: {sym} "
                    f"BUY ${amount_usd:.2f} @ ${new_price:,.4f} "
                    f"(score: {score:.2f}, trailing: {trailing_pct}%)"
                )

        # Log scan results
        self._save_autopilot_log({
            "type": "altcoin_hunter_scan",
            "symbols_scanned": scanned,
            "status": "opened_positions" if any(a["action"] == "BUY" for a in actions) else "rebalanced",
            "positions": max_positions - slots_open,
            "max_positions": max_positions,
            "deployed_usd": round(deployed, 2),
            "available_usd": round(available, 2),
            "top_scores": [
                {"symbol": s["symbol"], "score": s["score"]}
                for s in scored[:10]
            ],
            "actions": actions,
        })

    def _save_autopilot_log(self, entry: dict):
        """Append an entry to autopilot_log in portfolio (keep last 50)."""
        import time as _time
        entry["time"] = int(_time.time())
        pdata = self.portfolio._load_raw()
        log = pdata.get("autopilot_log", [])
        log.append(entry)
        log = log[-50:]
        pdata["autopilot_log"] = log
        path = os.path.join(self.portfolio.data_dir, self.portfolio.filename)
        import json as _json
        with open(path, "w") as f:
            _json.dump(pdata, f, indent=2)

    def _save_decision(self, decision: dict):
        """Save a structured decision to the decision log (keep last 100).

        Each decision captures WHY the bot acted (or didn't):
        - action: ENTER_LONG, ENTER_SHORT, EXIT, SKIP, ERROR
        - symbol, strategy_id, trade_mode
        - signal, confidence, reason (from strategy)
        - indicators: key technicals that drove the decision
        """
        import time as _time
        import json as _json
        decision["time"] = int(_time.time())
        decision.setdefault("source", "cycle")

        pdata = self.portfolio._load_raw()
        decisions = pdata.get("decision_log", [])
        decisions.append(decision)
        decisions = decisions[-100:]
        pdata["decision_log"] = decisions
        path = os.path.join(self.portfolio.data_dir, self.portfolio.filename)
        with open(path, "w") as f:
            _json.dump(pdata, f, indent=2)

    def _run_autopilot(self, prices: dict):
        """AI Autopilot — fully autonomous portfolio management.

        Scans all major symbols, picks the best opportunities using the
        full AI brain (technicals + whale + sentiment), and allocates
        the autopilot fund across up to 3 positions.

        The autopilot fund is stored in portfolio as 'autopilot_fund'.
        """
        pdata = self.portfolio._load_raw()
        autopilot = pdata.get("autopilot")
        if not autopilot or not autopilot.get("enabled"):
            return

        fund_total = autopilot.get("fund_usd", 0)
        if fund_total <= 0:
            return

        max_positions = autopilot.get("max_positions", 3)

        # Count existing autopilot positions
        autopilot_positions = [
            pos for pos in self.portfolio.positions.values()
            if pos.strategy_id == "autopilot"
        ]

        # Calculate how much is currently deployed vs available
        deployed = sum(pos.cost_usd for pos in autopilot_positions)
        available = fund_total - deployed
        slots_open = max_positions - len(autopilot_positions)

        if slots_open <= 0 or available < 10:
            self._save_autopilot_log({
                "type": "scan",
                "symbols_scanned": 0,
                "status": "fully_deployed" if slots_open <= 0 else "low_funds",
                "positions": len(autopilot_positions),
                "max_positions": max_positions,
                "deployed_usd": round(deployed, 2),
                "available_usd": round(available, 2),
                "candidates": [],
                "actions": [],
            })
            return

        # Scan all symbols for the best opportunity
        strategy = self._get_strategy("autopilot") or EnhancedEnsemble()
        candidates = []
        scanned = 0
        scan_details = []

        # Scan crypto symbols
        for sym in TICKER_SYMBOLS:
            try:
                if any(pos.symbol == sym for pos in autopilot_positions):
                    continue
                snap = self._get_snapshot(sym, "autopilot")
                prices[sym] = snap["price"]
                result = strategy.analyze(snap)
                scanned += 1
                scan_details.append({
                    "symbol": sym,
                    "price": round(snap["price"], 2),
                    "signal": result.signal.name,
                    "confidence": round(result.confidence, 3),
                    "whale": round(snap.get("whale_score", 0), 2),
                    "sentiment": round(snap.get("sentiment_score", 0), 2),
                    "reason": result.reason[:100],
                })
                if result.signal == Signal.BUY and result.confidence >= 0.55:
                    candidates.append((result.confidence, sym, snap, result))
            except Exception:
                continue

        # Scan stock symbols (if Alpaca is connected)
        if self.alpaca:
            for sym in STOCK_SYMBOLS:
                try:
                    if any(pos.symbol == sym for pos in autopilot_positions):
                        continue
                    snap = get_stock_snapshot(self.alpaca, sym)
                    prices[sym] = snap["price"]
                    result = strategy.analyze(snap)
                    scanned += 1
                    scan_details.append({
                        "symbol": sym,
                        "price": round(snap["price"], 2),
                        "signal": result.signal.name,
                        "confidence": round(result.confidence, 3),
                        "whale": 0.0,
                        "sentiment": 0.0,
                        "reason": result.reason[:100],
                    })
                    if result.signal == Signal.BUY and result.confidence >= 0.55:
                        candidates.append((result.confidence, sym, snap, result))
                except Exception:
                    continue

        # Save per-symbol decision reasoning for the autopilot scan
        for detail in scan_details:
            if detail["signal"] == "BUY" and detail["confidence"] >= 0.55:
                action = "CANDIDATE"
            else:
                action = "SKIP"
            self._save_decision({
                "action": action,
                "symbol": detail["symbol"],
                "strategy_id": "autopilot",
                "source": "autopilot",
                "trade_mode": self.mode,
                "signal": detail["signal"],
                "confidence": detail["confidence"],
                "reason": detail["reason"],
                "indicators": {
                    "price": detail["price"],
                    "whale_score": detail.get("whale", 0),
                    "sentiment": detail.get("sentiment", 0),
                },
            })

        if not candidates:
            self._save_autopilot_log({
                "type": "scan",
                "symbols_scanned": scanned,
                "status": "no_opportunity",
                "positions": len(autopilot_positions),
                "max_positions": max_positions,
                "deployed_usd": round(deployed, 2),
                "available_usd": round(available, 2),
                "candidates": [],
                "top_signals": sorted(scan_details, key=lambda x: x["confidence"], reverse=True)[:5],
                "actions": [],
            })
            return

        # Sort by confidence, pick the top N to fill available slots
        candidates.sort(reverse=True, key=lambda x: x[0])
        to_open = candidates[:slots_open]

        # Allocate funds evenly across new positions
        per_position = available / len(to_open)
        actions = []

        for confidence, sym, snap, result in to_open:
            new_price = snap["price"]
            amount_usd = min(per_position, available)
            if amount_usd < 10:
                break

            # Dynamic SL based on ATR
            atr = snap.get("atr", 0)
            if atr and new_price > 0:
                sl_pct = min(max((2 * atr / new_price) * 100, 1.0), 4.0)
            else:
                sl_pct = 2.5

            # Dynamic TP scaled by confidence
            tp_min_pct = 1.5 + confidence * 1.0  # 1.5% - 2.5%
            tp_max_pct = 5.0 + confidence * 5.0  # 5% - 10%

            limits = PositionLimits(
                stop_loss_price=new_price * (1 - sl_pct / 100),
                take_profit_min_price=new_price * (1 + tp_min_pct / 100),
                take_profit_max_price=new_price * (1 + tp_max_pct / 100),
                position_size_usd=amount_usd,
                position_size_asset=amount_usd / new_price,
                trailing_stop_pct=1.5,  # Autopilot always uses trailing stop
                highest_price_seen=new_price,
            )

            position = Position(
                symbol=sym, side="buy",
                entry_price=new_price,
                amount=limits.position_size_asset,
                cost_usd=amount_usd,
                limits=limits,
                strategy_reason=f"Autopilot AI: {result.reason[:120]}",
                strategy_id="autopilot",
                bet_mode="continuous",
                trade_mode=self.mode if self.mode == "live" else "paper",
                stop_loss_pct=sl_pct,
                tp_min_pct=tp_min_pct,
                tp_max_pct=tp_max_pct,
                trailing_stop_pct=1.5,
                highest_price_seen=new_price,
            )

            if self.mode == "live":
                try:
                    if is_stock_symbol(sym) and self.alpaca:
                        from vesper.exchange import alpaca_market_buy
                        order = alpaca_market_buy(self.alpaca, sym, amount_usd)
                        position.entry_price = float(order.get("average", new_price))
                        position.amount = float(order.get("filled", limits.position_size_asset))
                        position.cost_usd = position.entry_price * position.amount
                    else:
                        from vesper.exchange import place_market_buy
                        order = place_market_buy(self.exchange, sym, limits.position_size_asset)
                        position.entry_price = float(order.get("average", new_price))
                        position.amount = float(order.get("filled", limits.position_size_asset))
                        position.cost_usd = position.entry_price * position.amount
                except Exception as e:
                    self.logger.error(f"[User:{self.email}] Autopilot BUY failed {sym}: {e}")
                    continue

            if self.portfolio.open_position(position):
                available -= amount_usd
                actions.append({
                    "action": "BUY",
                    "symbol": sym,
                    "amount_usd": round(amount_usd, 2),
                    "price": round(new_price, 2),
                    "confidence": round(confidence, 3),
                    "whale": round(snap.get("whale_score", 0), 2),
                    "sentiment": round(snap.get("sentiment_score", 0), 2),
                    "reason": result.reason[:120],
                })
                self._save_decision({
                    "action": "ENTER_LONG",
                    "symbol": sym,
                    "strategy_id": "autopilot",
                    "source": "autopilot",
                    "trade_mode": self.mode if self.mode == "live" else "paper",
                    "signal": "BUY",
                    "confidence": round(confidence, 3),
                    "reason": result.reason,
                    "amount_usd": round(amount_usd, 2),
                    "indicators": self._snapshot_indicators(snap),
                })
                self.logger.info(
                    f"[User:{self.email}] AUTOPILOT: {sym} "
                    f"BUY ${amount_usd:.2f} @ ${new_price:,.2f} "
                    f"(AI confidence: {confidence:.0%}, "
                    f"whale: {snap.get('whale_score', 0):+.2f}, "
                    f"sentiment: {snap.get('sentiment_score', 0):+.2f})"
                )

        # Log the scan results
        candidate_info = [{"symbol": s, "confidence": round(c, 3)} for c, s, _, _ in candidates]
        self._save_autopilot_log({
            "type": "scan",
            "symbols_scanned": scanned,
            "status": "opened_positions" if actions else "no_opportunity",
            "positions": len(autopilot_positions) + len(actions),
            "max_positions": max_positions,
            "deployed_usd": round(deployed + sum(a["amount_usd"] for a in actions), 2),
            "available_usd": round(available, 2),
            "candidates": candidate_info,
            "top_signals": sorted(scan_details, key=lambda x: x["confidence"], reverse=True)[:5],
            "actions": actions,
        })


class Vesper:
    """Main orchestrator — manages all user bots."""

    def __init__(self):
        self.logger = setup_logger()

    def run_all_users(self):
        """Run a trading cycle for every active user."""
        from vesper.dashboard.database import get_active_users, init_db
        init_db()

        active_users = get_active_users()
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Trading cycle — {len(active_users)} active user(s)")

        for user in active_users:
            try:
                exchange_cfg = ExchangeConfig(
                    api_key=user.get_api_key(),
                    api_secret=user.get_api_secret(),
                )
                risk_cfg = RiskConfig(
                    stop_loss_pct=user.stop_loss_pct,
                    take_profit_min_pct=user.take_profit_min_pct,
                    take_profit_max_pct=user.take_profit_max_pct,
                    max_position_pct=user.max_position_pct,
                )
                symbols = [s.strip() for s in user.symbols.split(",")]

                # Create Alpaca client if user has Alpaca keys
                alpaca_client = None
                if user.has_alpaca:
                    try:
                        alpaca_client = AlpacaExchange(
                            api_key=user.get_alpaca_key(),
                            api_secret=user.get_alpaca_secret(),
                            paper=(user.trading_mode != "live"),
                        )
                    except Exception as e:
                        self.logger.error(f"[User:{user.email}] Alpaca init failed: {e}")

                bot = UserBot(
                    user_id=user.id,
                    email=user.email,
                    exchange_cfg=exchange_cfg,
                    risk_cfg=risk_cfg,
                    symbols=symbols,
                    mode=user.trading_mode,
                    paper_balance=user.paper_balance,
                    logger=self.logger,
                    alpaca_client=alpaca_client,
                )
                bot.run_cycle()
            except Exception as e:
                self.logger.error(f"[User:{user.email}] Cycle failed: {e}")

    def start(self):
        console.print(f"\n[bold magenta]{'='*60}[/]")
        console.print(f"[bold magenta]  VESPER — Crypto Trading Platform v0.2.0[/]")
        console.print(f"[bold magenta]{'='*60}[/]\n")

        # Start dashboard
        self._start_dashboard()

        # Run first cycle
        self.run_all_users()

        # Schedule cycles every minute (bot checks each user's interval)
        scheduler = BlockingScheduler()
        scheduler.add_job(
            self.run_all_users,
            "interval",
            minutes=1,
            id="trading_cycle",
        )

        def shutdown(signum, frame):
            self.logger.info("Shutting down Vesper...")
            scheduler.shutdown(wait=False)

        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)

        self.logger.info("Scheduler started — checking users every minute")
        self.logger.info("Dashboard running at http://0.0.0.0:8080")
        scheduler.start()

    def _start_dashboard(self):
        from vesper.dashboard.app import app as dashboard_app

        def run():
            try:
                uvicorn.run(dashboard_app, host="0.0.0.0", port=8080, log_level="info")
            except Exception as e:
                self.logger.error(f"Dashboard failed to start: {e}")

        threading.Thread(target=run, daemon=True).start()


def main():
    bot = Vesper()
    bot.start()


if __name__ == "__main__":
    main()
