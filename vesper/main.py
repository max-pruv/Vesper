"""Vesper — Main orchestrator and scheduler for multi-user trading."""

import sys
import os
import signal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading

from apscheduler.schedulers.blocking import BlockingScheduler
import uvicorn

from config.settings import ExchangeConfig, RiskConfig, TICKER_SYMBOLS, STOCK_SYMBOLS, ALL_CRYPTO_SYMBOLS, is_stock_symbol
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

        summary = self.portfolio.summary(prices)
        self.logger.info(
            f"[User:{self.email}] Portfolio: ${summary['total_value']:,.2f} | "
            f"P&L: ${summary['total_pnl_usd']:+,.2f} ({summary['total_pnl_pct']:+.2f}%)"
        )

        # Record portfolio value snapshot for the equity chart
        try:
            self.portfolio.record_value_snapshot(prices)
        except Exception:
            pass

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
                if existing_pos.bet_mode == "continuous":
                    closed_continuous = existing_pos
                self._close_position(existing_pos, snapshot["price"], reason)
                if closed_continuous is None:
                    return  # One-off: done
            else:
                # ── AI-driven exit: check if AI recommends closing ──
                # The AI can detect whale dumping / bearish sentiment shift
                # before the stop-loss is hit, protecting profits
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
            return

        if not self.risk_manager.can_open_position(len(self.portfolio.positions)):
            return

        if result.signal == Signal.BUY:
            if closed_continuous:
                self._reopen_continuous(closed_continuous, snapshot)
            else:
                self._open_position(symbol, snapshot, result)
        elif result.signal == Signal.SHORT:
            # Only short stocks via Alpaca (crypto shorting not supported)
            if is_stock_symbol(symbol) and self.alpaca:
                self._open_short_position(symbol, snapshot, result)
            else:
                self.logger.debug(
                    f"[User:{self.email}] SHORT signal for {symbol} — "
                    f"skipping (shorting only available for stocks via Alpaca)"
                )

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

    def _open_short_position(self, symbol, snapshot, result):
        """Open a short position (sell to open). Currently stocks only via Alpaca."""
        portfolio_value = self.portfolio.total_value({symbol: snapshot["price"]})
        limits = self.risk_manager.calculate_position(
            entry_price=snapshot["price"],
            portfolio_value=portfolio_value,
            atr=snapshot["atr"],
            confidence=result.confidence,
            side="sell",
        )

        position = Position(
            symbol=symbol, side="sell",
            entry_price=snapshot["price"],
            amount=limits.position_size_asset,
            cost_usd=limits.position_size_usd,
            limits=limits,
            strategy_reason=f"SHORT: {result.reason}",
        )

        if self.mode == "paper":
            if self.portfolio.open_position(position):
                self.logger.info(
                    f"[User:{self.email}] PAPER SHORT {symbol}: "
                    f"{position.amount:.6f} @ ${position.entry_price:,.2f}"
                )
        else:
            try:
                from vesper.exchange import alpaca_market_sell
                order = alpaca_market_sell(self.alpaca, symbol, limits.position_size_asset)
                position.entry_price = float(order.get("average", snapshot["price"]))
                position.amount = float(order.get("filled", limits.position_size_asset))
                position.cost_usd = position.entry_price * position.amount
                self.portfolio.open_position(position)
                self.logger.info(f"[User:{self.email}] LIVE SHORT {symbol}: filled")
            except Exception as e:
                self.logger.error(f"[User:{self.email}] SHORT failed {symbol}: {e}")

    def _close_position(self, position: Position, exit_price: float, reason: str):
        trade_mode = position.trade_mode if position.trade_mode else self.mode
        if trade_mode == "real" or self.mode == "live":
            try:
                if position.side == "sell":
                    # Closing a short = buy back
                    if is_stock_symbol(position.symbol) and self.alpaca:
                        from vesper.exchange import alpaca_market_buy
                        alpaca_market_buy(self.alpaca, position.symbol, position.cost_usd)
                    else:
                        from vesper.exchange import place_market_buy
                        place_market_buy(self.exchange, position.symbol, position.amount)
                else:
                    # Closing a long = sell
                    if is_stock_symbol(position.symbol) and self.alpaca:
                        from vesper.exchange import alpaca_market_sell
                        alpaca_market_sell(self.alpaca, position.symbol, position.amount)
                    else:
                        from vesper.exchange import place_market_sell
                        place_market_sell(self.exchange, position.symbol, position.amount)
            except Exception as e:
                self.logger.error(f"[User:{self.email}] Close failed: {e}")
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

        # Scan all crypto symbols (core + altcoins)
        for sym in ALL_CRYPTO_SYMBOLS:
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
                    # Accept both BUY and SHORT signals for stocks (Alpaca supports shorting)
                    if result.signal in (Signal.BUY, Signal.SHORT) and result.confidence >= 0.55:
                        candidates.append((result.confidence, sym, snap, result))
                except Exception:
                    continue

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

        # Sort by confidence + whale score for a composite ranking
        def _rank(c):
            conf, sym, snap, result = c
            whale = snap.get("whale_score", 0)
            return conf * 0.7 + max(whale, 0) * 0.3  # confidence dominates, whale is a bonus
        candidates.sort(reverse=True, key=_rank)
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

            # Determine side (long or short)
            side = "sell" if result.signal == Signal.SHORT else "buy"

            if side == "buy":
                limits = PositionLimits(
                    stop_loss_price=new_price * (1 - sl_pct / 100),
                    take_profit_min_price=new_price * (1 + tp_min_pct / 100),
                    take_profit_max_price=new_price * (1 + tp_max_pct / 100),
                    position_size_usd=amount_usd,
                    position_size_asset=amount_usd / new_price,
                    trailing_stop_pct=1.5,
                    highest_price_seen=new_price,
                )
            else:
                # SHORT: SL above entry, TP below entry (inverted)
                limits = PositionLimits(
                    stop_loss_price=new_price * (1 + sl_pct / 100),
                    take_profit_min_price=new_price * (1 - tp_min_pct / 100),
                    take_profit_max_price=new_price * (1 - tp_max_pct / 100),
                    position_size_usd=amount_usd,
                    position_size_asset=amount_usd / new_price,
                    trailing_stop_pct=1.5,
                    highest_price_seen=new_price,
                )

            position = Position(
                symbol=sym, side=side,
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
                    if side == "sell":
                        # SHORT entry — sell to open
                        from vesper.exchange import alpaca_market_sell
                        order = alpaca_market_sell(self.alpaca, sym, limits.position_size_asset)
                    elif is_stock_symbol(sym) and self.alpaca:
                        from vesper.exchange import alpaca_market_buy
                        order = alpaca_market_buy(self.alpaca, sym, amount_usd)
                    else:
                        from vesper.exchange import place_market_buy
                        order = place_market_buy(self.exchange, sym, limits.position_size_asset)
                    position.entry_price = float(order.get("average", new_price))
                    position.amount = float(order.get("filled", limits.position_size_asset))
                    position.cost_usd = position.entry_price * position.amount
                except Exception as e:
                    self.logger.error(f"[User:{self.email}] Autopilot {side.upper()} failed {sym}: {e}")
                    continue

            if self.portfolio.open_position(position):
                available -= amount_usd
                actions.append({
                    "action": "SHORT" if side == "sell" else "BUY",
                    "symbol": sym,
                    "amount_usd": round(amount_usd, 2),
                    "price": round(new_price, 2),
                    "confidence": round(confidence, 3),
                    "whale": round(snap.get("whale_score", 0), 2),
                    "sentiment": round(snap.get("sentiment_score", 0), 2),
                    "reason": result.reason[:120],
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
            uvicorn.run(dashboard_app, host="0.0.0.0", port=8080, log_level="warning")

        threading.Thread(target=run, daemon=True).start()


def main():
    bot = Vesper()
    bot.start()


if __name__ == "__main__":
    main()
