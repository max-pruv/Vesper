"""Vesper — Main orchestrator and scheduler for multi-user trading."""

import os
import signal
import threading

import ccxt

from apscheduler.schedulers.blocking import BlockingScheduler
import uvicorn

from config.settings import ExchangeConfig, RiskConfig, TICKER_SYMBOLS, STOCK_SYMBOLS, ALTCOIN_UNIVERSE, is_stock_symbol
from vesper.exchange import create_exchange, AlpacaExchange
from vesper.market_data import (
    get_market_snapshot, get_multi_tf_snapshot,
    get_order_book_pressure, fetch_fear_greed,
    enrich_with_intelligence, get_stock_snapshot,
    discover_trending_coins,
)
from vesper.strategies import Signal
from vesper.strategies.catalog import STRATEGY_MAP
from vesper.strategies.ensemble import EnhancedEnsemble
from vesper.risk import RiskManager, PositionLimits
from vesper.portfolio import Portfolio, Position
from vesper.logger import setup_logger, console

# Thread-local storage for per-thread ccxt exchange instances
_thread_local = threading.local()
# Shared exchange state (resolved once at startup, reused everywhere)
_exchange_lock = threading.Lock()
_exchange_name: str | None = None
_exchange_factory = None  # callable() -> ccxt.Exchange
_markets_data: dict | None = None

# Exchanges to try, in preference order (best USDT pair coverage first)
_EXCHANGE_CANDIDATES = [
    ("binance", lambda: ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})),
    ("binanceus", lambda: ccxt.binanceus({"enableRateLimit": True})),
    ("kucoin", lambda: ccxt.kucoin({"enableRateLimit": True})),
    ("okx", lambda: ccxt.okx({"enableRateLimit": True})),
    ("bybit", lambda: ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "spot"}})),
]


def _resolve_exchange(logger=None) -> tuple:
    """Find the first reachable crypto exchange and cache its markets.

    Tries multiple exchanges in order until one responds with OHLCV data.
    Returns (exchange_instance, exchange_name).
    """
    global _exchange_name, _exchange_factory, _markets_data
    with _exchange_lock:
        if _markets_data is not None:
            # Already resolved — create a new instance with cached markets
            ex = _exchange_factory()
            ex.markets = _markets_data["markets"]
            ex.markets_by_id = _markets_data["markets_by_id"]
            ex.currencies = _markets_data["currencies"]
            ex.currencies_by_id = _markets_data["currencies_by_id"]
            return ex, _exchange_name

        # First time — probe exchanges
        for name, factory in _EXCHANGE_CANDIDATES:
            try:
                ex = factory()
                ex.load_markets()
                # Verify OHLCV works with a quick BTC fetch
                ohlcv = ex.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=5)
                if not ohlcv:
                    continue
                # Success — cache this exchange
                _exchange_name = name
                _exchange_factory = factory
                _markets_data = {
                    "markets": ex.markets,
                    "markets_by_id": ex.markets_by_id,
                    "currencies": ex.currencies,
                    "currencies_by_id": ex.currencies_by_id,
                }
                if logger:
                    logger.info(f"[exchange] Using {name} for market data ({len(ex.markets)} markets)")
                return ex, name
            except Exception as e:
                if logger:
                    logger.warning(f"[exchange] {name} failed: {str(e)[:100]}")
                continue

        raise RuntimeError("No crypto exchange reachable from this server")


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
        # Public exchange for market data — auto-selects reachable exchange
        self.public_exchange, self._exchange_name = _resolve_exchange(logger)
        self.alpaca = alpaca_client  # None if user hasn't connected Alpaca

    def _get_thread_exchange(self) -> ccxt.Exchange:
        """Get a thread-local exchange instance (thread-safe for parallel scanning)."""
        ex = getattr(_thread_local, "data_exchange", None)
        if ex is None:
            ex, _ = _resolve_exchange()
            _thread_local.data_exchange = ex
        return ex

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

        # Predictions Autopilot: AI research on prediction markets, auto-betting
        self._run_predictions_autopilot()

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

    def _get_snapshot(self, symbol: str, strategy_id: str, exchange_override=None) -> dict:
        """Fetch the right market snapshot for a strategy's timeframe.

        Uses public_exchange (Binance) for market data — reliable USDT pairs.
        Pass exchange_override for thread-safe parallel scanning.
        """
        # Stock symbols use Alpaca data
        if is_stock_symbol(symbol) and self.alpaca:
            return get_stock_snapshot(self.alpaca, symbol)

        # Use override (thread-local) or default public exchange
        data_exchange = exchange_override or self.public_exchange

        config = STRATEGY_MAP.get(strategy_id, {})
        timeframe = config.get("timeframe", "1h")

        if timeframe == "multi":
            # Multi-timeframe: 1h + 4h with alignment
            snapshot = get_multi_tf_snapshot(data_exchange, symbol)
            # Enrich with order book + sentiment for enhanced strategies
            try:
                ob = get_order_book_pressure(data_exchange, symbol)
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
                enrich_with_intelligence(data_exchange, symbol, snapshot)
            except Exception:
                pass
        elif timeframe:
            snapshot = get_market_snapshot(data_exchange, symbol, timeframe=timeframe)
        else:
            snapshot = get_market_snapshot(data_exchange, symbol)

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
            trade_mode="real" if self.mode == "live" else "paper",
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

        max_positions = hunter_cfg.get("max_positions", 20)
        trailing_pct = hunter_cfg.get("trailing_stop_pct", 2.0)

        # Risk tolerance controls thresholds
        risk_level = hunter_cfg.get("risk_level", "aggressive")
        score_thresholds = {"conservative": 0.60, "moderate": 0.50, "aggressive": 0.40}
        min_score = score_thresholds.get(risk_level, 0.50)

        self.logger.info(
            f"[User:{self.email}] [altcoin_hunter] Starting scan: "
            f"fund=${fund_total:.2f}, max_positions={max_positions}, "
            f"risk={risk_level}, min_score={min_score}"
        )

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

        # ── Phase 2: Score all altcoins (static list + dynamic discovery) ──
        from vesper.strategies.altcoin_hunter import compute_trend_score

        # Merge static universe with dynamically discovered trending coins
        try:
            trending = discover_trending_coins(self.public_exchange)
        except Exception as e:
            self.logger.warning(f"[altcoin_hunter] discover_trending_coins failed: {e}")
            trending = []
        scan_universe = list(dict.fromkeys(ALTCOIN_UNIVERSE + trending))  # dedupe, preserve order
        self.logger.info(f"[altcoin_hunter] Scanning {len(scan_universe)} coins ({len(trending)} trending)")

        scored = []
        scanned = 0

        # Parallel scan: fetch & score coins concurrently (5 threads)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _scan_coin(sym):
            if sym == "BTC/USDT":
                return None
            try:
                thread_ex = self._get_thread_exchange()
                snap = self._get_snapshot(sym, "altcoin_hunter", exchange_override=thread_ex)
                score_data = compute_trend_score(snap, btc_snapshot)
                return {
                    "symbol": sym,
                    "score": score_data["score"],
                    "signal": score_data["signal"],
                    "confidence": score_data["confidence"],
                    "factors": score_data["factors"],
                    "snapshot": snap,
                    "price": snap["price"],
                }
            except Exception as e:
                self.logger.warning(f"[scan] {sym} failed: {e}")
                return None

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(_scan_coin, sym): sym for sym in scan_universe}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    scanned += 1
                    prices[result["symbol"]] = result["price"]
                    scored.append(result)

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
        self.logger.info(
            f"[User:{self.email}] [altcoin_hunter] Phase 4: "
            f"scanned={scanned}, slots_open={slots_open}, "
            f"available=${available:.2f}, scored_above_0.4={sum(1 for s in scored if s['score'] >= 0.4)}"
        )
        if slots_open <= 0 or available < 10:
            self.logger.info(
                f"[User:{self.email}] [altcoin_hunter] Skipping phase 4: "
                f"{'fully_deployed' if slots_open <= 0 else 'low_funds'}"
            )
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

        # Filter candidates based on risk level
        held_symbols = {pos.symbol for pos in self.portfolio.positions.values()}
        if risk_level == "aggressive":
            # Aggressive: accept any signal above score threshold (incl. HOLD)
            candidates = [
                s for s in scored
                if s["score"] >= min_score
                and s["signal"] != Signal.SELL
                and s["symbol"] not in held_symbols
            ]
        else:
            # Moderate/Conservative: require explicit BUY signal
            candidates = [
                s for s in scored
                if s["score"] >= min_score
                and s["signal"] == Signal.BUY
                and s["symbol"] not in held_symbols
            ]

        # Deep research on top candidates (max 3 to control costs)
        top5 = [(c['symbol'], round(c['score'], 2)) for c in candidates[:5]]
        self.logger.info(
            f"[User:{self.email}] [altcoin_hunter] {len(candidates)} candidates "
            f"above min_score={min_score}: {top5}"
        )
        from vesper.ai_research import research_asset
        for c in candidates[:3]:
            try:
                self.logger.info(
                    f"[User:{self.email}] [altcoin_hunter] Researching {c['symbol']} "
                    f"(score={c['score']:.2f})"
                )
                research = research_asset(
                    c["symbol"],
                    c["factors"],
                    c["snapshot"]["price"],
                    asset_type="crypto",
                )
                c["deep_research"] = research
                self.logger.info(
                    f"[User:{self.email}] [altcoin_hunter] Research result for {c['symbol']}: "
                    f"researched={research.get('researched')}, signal={research.get('signal')}"
                )
                # Boost score if deep research agrees
                if research.get("researched") and research.get("signal") == "BUY":
                    c["score"] = min(1.0, c["score"] + research["confidence"] * 0.15)
                    c["confidence"] = min(1.0, c.get("confidence", 0.5) + 0.10)
                elif research.get("researched") and research.get("signal") == "SELL":
                    c["score"] = max(0.0, c["score"] - 0.20)
            except Exception as e:
                self.logger.warning(
                    f"[User:{self.email}] [altcoin_hunter] Research failed for {c['symbol']}: {e}"
                )
                c["deep_research"] = {}

        # Re-sort after research adjustments
        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = [c for c in candidates if c["score"] >= min_score]

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

            # Build deep reasoning string
            deep = candidate.get("deep_research", {})
            reason_parts = [f"Altcoin Hunter: score {score:.2f}"]
            reason_parts.append(" | ".join(f"{k}={v:.2f}" for k, v in candidate["factors"].items()))
            if deep.get("researched"):
                reason_parts.append(f"Deep Search ({deep.get('engine', 'AI')}): {deep.get('reasoning', '')[:200]}")
                if deep.get("bullish_factors"):
                    reason_parts.append("Bullish: " + "; ".join(deep["bullish_factors"][:3]))
                if deep.get("bearish_factors"):
                    reason_parts.append("Bearish: " + "; ".join(deep["bearish_factors"][:3]))

            position = Position(
                symbol=sym, side="buy",
                entry_price=new_price,
                amount=limits.position_size_asset,
                cost_usd=amount_usd,
                limits=limits,
                strategy_reason=" | ".join(reason_parts),
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
                # Store full analysis data for "Learn More"
                analysis_data = {
                    "indicators": self._snapshot_indicators(snap),
                    "trend_factors": candidate["factors"],
                    "trend_score": score,
                    "deep_research": {
                        "signal": deep.get("signal", ""),
                        "confidence": deep.get("confidence", 0),
                        "reasoning": deep.get("reasoning", ""),
                        "bullish_factors": deep.get("bullish_factors", []),
                        "bearish_factors": deep.get("bearish_factors", []),
                        "catalysts": deep.get("catalysts", []),
                        "sources": deep.get("sources", []),
                        "search_summary": deep.get("search_summary", ""),
                        "engine": deep.get("engine", ""),
                    },
                    "risk_level": risk_level,
                }
                self.portfolio._save_position_analysis(position.id, analysis_data)

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

    def _run_predictions_autopilot(self):
        """Predictions Autopilot — AI-driven prediction market auto-investing.

        Scans trending prediction markets, uses Perplexity + Claude deep research
        to find mispricings, then auto-invests in markets with significant AI edge.

        Rate-limited to once per hour to control AI research costs.
        Cost: ~$0.30-0.50/day (3-5 markets researched per cycle, 4h cache).
        """
        import time as _time

        pdata = self.portfolio._load_raw()
        pred_cfg = pdata.get("predictions_autopilot")
        if not pred_cfg or not pred_cfg.get("enabled"):
            return

        fund_total = pred_cfg.get("fund_usd", 0)
        if fund_total <= 0:
            return

        # Rate limit: run predictions scan every 60 minutes
        last_scan = pred_cfg.get("last_scan_time", 0)
        if _time.time() - last_scan < 3600:
            return

        max_positions = pred_cfg.get("max_positions", 20)
        min_edge = pred_cfg.get("min_edge_pct", 5.0)

        # Collect existing prediction positions
        pred_positions = [
            pos for pos in self.portfolio.positions.values()
            if pos.strategy_id == "predictions"
        ]

        deployed = sum(pos.cost_usd for pos in pred_positions)
        available = fund_total - deployed
        slots_open = max_positions - len(pred_positions)

        # Update last scan time now (even if we don't find anything)
        pred_cfg["last_scan_time"] = _time.time()
        pdata["predictions_autopilot"] = pred_cfg

        if slots_open <= 0 or available < 5:
            self._save_autopilot_log({
                "type": "predictions_scan",
                "markets_scanned": 0,
                "status": "fully_deployed" if slots_open <= 0 else "low_funds",
                "positions": len(pred_positions),
                "max_positions": max_positions,
                "deployed_usd": round(deployed, 2),
                "available_usd": round(available, 2),
                "actions": [],
            })
            # Save updated last_scan_time
            path = os.path.join(self.portfolio.data_dir, self.portfolio.filename)
            import json as _json
            with open(path, "w") as f:
                _json.dump(pdata, f, indent=2)
            return

        # Fetch trending prediction markets from all sources
        markets = []
        try:
            from vesper.polymarket import get_trending_markets
            poly_markets = get_trending_markets(limit=30, max_days=14)
            for m in poly_markets:
                m["source"] = "polymarket"
            markets.extend(poly_markets)
        except Exception as e:
            self.logger.warning(f"[User:{self.email}] Predictions: Polymarket fetch failed: {e}")

        try:
            from vesper.kalshi import get_kalshi_markets
            kalshi_markets = get_kalshi_markets(limit=20, max_days=14)
            for m in kalshi_markets:
                m["source"] = "kalshi"
            markets.extend(kalshi_markets)
        except Exception as e:
            self.logger.warning(f"[User:{self.email}] Predictions: Kalshi fetch failed: {e}")

        if not markets:
            self._save_autopilot_log({
                "type": "predictions_scan",
                "markets_scanned": 0,
                "status": "no_markets",
                "positions": len(pred_positions),
                "max_positions": max_positions,
                "deployed_usd": round(deployed, 2),
                "available_usd": round(available, 2),
                "actions": [],
            })
            path = os.path.join(self.portfolio.data_dir, self.portfolio.filename)
            import json as _json
            with open(path, "w") as f:
                _json.dump(pdata, f, indent=2)
            return

        # Pre-filter: markets with decent volume (let AI research decide on edge)
        candidates = [
            m for m in markets
            if m.get("volume", 0) > 500
        ]

        # Skip markets we already have positions on
        held_questions = {
            pos.prediction_question
            for pos in pred_positions
            if hasattr(pos, "prediction_question") and pos.prediction_question
        }
        # Also check raw pdata for prediction_question field
        for pos in pred_positions:
            raw_pos = pdata.get("positions", {}).get(pos.id, {})
            q = raw_pos.get("prediction_question", "")
            if q:
                held_questions.add(q)

        candidates = [
            m for m in candidates
            if m.get("question", "") not in held_questions
        ]

        # Deep AI research on top 3 candidates (cost control)
        research_limit = min(3, len(candidates))
        researched = []
        scanned_count = len(markets)
        self.logger.info(
            f"[User:{self.email}] [predictions] {len(markets)} markets scanned, "
            f"{len(candidates)} candidates after filters, researching top {research_limit}"
        )

        try:
            from vesper.ai_research import research_market
        except ImportError:
            self.logger.error(f"[User:{self.email}] Predictions: ai_research module not available")
            return

        for market in candidates[:research_limit]:
            try:
                question = market.get("question", "")
                mkt_prob = market.get("market_probability", 50)
                category = (market.get("tags") or [""])[0]

                research = research_market(
                    question=question,
                    market_probability=mkt_prob,
                    category=category,
                )

                if not research.get("researched"):
                    continue

                ai_prob = research.get("probability", mkt_prob)
                edge = abs(ai_prob - mkt_prob)

                self._save_decision({
                    "action": "CANDIDATE" if edge >= min_edge else "SKIP",
                    "symbol": question[:40],
                    "strategy_id": "predictions",
                    "source": "predictions_autopilot",
                    "trade_mode": "paper",
                    "confidence": round(edge / 100, 3),
                    "reason": (
                        f"AI: {ai_prob}% vs Market: {mkt_prob}% "
                        f"(edge {edge:.1f}%) — {research.get('reasoning', '')[:100]}"
                    ),
                })

                if edge >= min_edge:
                    researched.append({
                        "market": market,
                        "research": research,
                        "edge": edge,
                        "ai_prob": ai_prob,
                        "mkt_prob": mkt_prob,
                    })
            except Exception as e:
                self.logger.warning(f"[User:{self.email}] Predictions research error: {e}")
                continue

        # Place paper bets on researched markets with sufficient edge
        actions = []
        for item in researched[:slots_open]:
            market = item["market"]
            research = item["research"]
            ai_prob = item["ai_prob"]
            mkt_prob = item["mkt_prob"]
            edge = item["edge"]

            amount_usd = min(available / max(slots_open, 1), available)
            if amount_usd < 5:
                break

            question = market.get("question", "")
            # Bet YES if AI thinks higher probability, NO if lower
            pred_side = "yes" if ai_prob > mkt_prob else "no"
            # Entry "price" = cost per share (e.g., YES at 65% costs $0.65/share)
            share_price = (mkt_prob / 100) if pred_side == "yes" else (1 - mkt_prob / 100)
            share_price = max(share_price, 0.01)
            num_shares = amount_usd / share_price

            # For predictions, SL/TP are based on probability shifts
            # SL: if probability moves 15% against us, cut
            # TP: if probability moves to 90%+ in our favor, take profit
            if pred_side == "yes":
                sl_price = max(0.05, share_price * 0.6)  # 40% drop in prob price
                tp_price = min(0.95, share_price * 1.5)   # 50% gain in prob price
            else:
                sl_price = max(0.05, share_price * 0.6)
                tp_price = min(0.95, share_price * 1.5)

            pos_id = f"PRED-{int(_time.time())}-{len(actions)}"

            limits = PositionLimits(
                stop_loss_price=sl_price,
                take_profit_min_price=tp_price * 0.8,
                take_profit_max_price=tp_price,
                position_size_usd=amount_usd,
                position_size_asset=num_shares,
            )

            position = Position(
                symbol=f"PRED:{question[:50]}",
                side="buy",
                entry_price=share_price,
                amount=num_shares,
                cost_usd=amount_usd,
                limits=limits,
                strategy_reason=(
                    f"Predictions AI: {pred_side.upper()} {question[:60]} "
                    f"(AI {ai_prob}% vs Mkt {mkt_prob}%, edge {edge:.1f}%)"
                ),
                strategy_id="predictions",
                bet_mode="one_off",
                trade_mode="paper",
                stop_loss_pct=40.0,
                tp_min_pct=40.0,
                tp_max_pct=50.0,
            )

            if self.portfolio.open_position(position):
                available -= amount_usd
                slots_open -= 1
                actions.append({
                    "action": "BUY",
                    "question": question[:80],
                    "side": pred_side,
                    "amount_usd": round(amount_usd, 2),
                    "ai_prob": ai_prob,
                    "mkt_prob": mkt_prob,
                    "edge": round(edge, 1),
                })

                # Also store prediction-specific fields in the raw position
                raw_pdata = self.portfolio._load_raw()
                if position.id in raw_pdata.get("positions", {}):
                    raw_pdata["positions"][position.id]["prediction_question"] = question
                    raw_pdata["positions"][position.id]["prediction_side"] = pred_side
                    raw_pdata["positions"][position.id]["prediction_ai_prob"] = ai_prob
                    raw_pdata["positions"][position.id]["prediction_mkt_prob"] = mkt_prob
                    raw_pdata["positions"][position.id]["prediction_edge"] = round(edge, 1)
                    path = os.path.join(self.portfolio.data_dir, self.portfolio.filename)
                    import json as _json
                    with open(path, "w") as f:
                        _json.dump(raw_pdata, f, indent=2)

                self._save_decision({
                    "action": "ENTER_LONG",
                    "symbol": f"PRED:{question[:40]}",
                    "strategy_id": "predictions",
                    "source": "predictions_autopilot",
                    "trade_mode": "paper",
                    "signal": "BUY",
                    "confidence": round(edge / 100, 3),
                    "reason": (
                        f"{pred_side.upper()} @ {share_price:.2f} — "
                        f"AI {ai_prob}% vs Mkt {mkt_prob}% — "
                        f"{research.get('reasoning', '')[:80]}"
                    ),
                    "amount_usd": round(amount_usd, 2),
                })

                self.logger.info(
                    f"[User:{self.email}] PREDICTIONS: {pred_side.upper()} "
                    f"'{question[:50]}' ${amount_usd:.2f} "
                    f"(AI: {ai_prob}% vs Market: {mkt_prob}%, edge: {edge:.1f}%)"
                )

        # Log scan results
        self._save_autopilot_log({
            "type": "predictions_scan",
            "markets_scanned": scanned_count,
            "researched": research_limit,
            "status": "invested" if actions else "no_edge",
            "positions": max_positions - slots_open,
            "max_positions": max_positions,
            "deployed_usd": round(deployed + sum(a.get("amount_usd", 0) for a in actions), 2),
            "available_usd": round(available, 2),
            "actions": actions,
        })

        # Make sure last_scan_time is persisted
        final_pdata = self.portfolio._load_raw()
        if "predictions_autopilot" not in final_pdata:
            final_pdata["predictions_autopilot"] = pred_cfg
        else:
            final_pdata["predictions_autopilot"]["last_scan_time"] = _time.time()
        path = os.path.join(self.portfolio.data_dir, self.portfolio.filename)
        import json as _json
        with open(path, "w") as f:
            _json.dump(final_pdata, f, indent=2)

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

        max_positions = autopilot.get("max_positions", 20)

        # Risk tolerance
        risk_level = autopilot.get("risk_level", "aggressive")
        ensemble_conf = {"conservative": 0.45, "moderate": 0.30, "aggressive": 0.15}
        entry_conf = {"conservative": 0.55, "moderate": 0.35, "aggressive": 0.20}
        min_ensemble_conf = ensemble_conf.get(risk_level, 0.30)
        min_entry_conf = entry_conf.get(risk_level, 0.35)

        # Count existing autopilot positions
        autopilot_positions = [
            pos for pos in self.portfolio.positions.values()
            if pos.strategy_id == "autopilot"
        ]

        # Calculate how much is currently deployed vs available
        deployed = sum(pos.cost_usd for pos in autopilot_positions)
        available = fund_total - deployed
        slots_open = max_positions - len(autopilot_positions)

        self.logger.info(
            f"[User:{self.email}] [autopilot] Starting scan: "
            f"fund=${fund_total:.2f}, deployed=${deployed:.2f}, "
            f"available=${available:.2f}, slots={slots_open}/{max_positions}, risk={risk_level}"
        )

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
        strategy = EnhancedEnsemble(min_confidence=min_ensemble_conf)
        candidates = []
        scanned = 0
        scan_details = []

        # Parallel scan: crypto + stock symbols concurrently
        from concurrent.futures import ThreadPoolExecutor, as_completed
        held = {pos.symbol for pos in autopilot_positions}

        def _ap_scan_crypto(sym):
            if sym in held:
                return None
            try:
                thread_ex = self._get_thread_exchange()
                snap = self._get_snapshot(sym, "autopilot", exchange_override=thread_ex)
                result = strategy.analyze(snap)
                return {
                    "symbol": sym, "price": snap["price"], "snap": snap,
                    "result": result, "signal": result.signal.name,
                    "confidence": result.confidence,
                    "whale": snap.get("whale_score", 0),
                    "sentiment": snap.get("sentiment_score", 0),
                    "reason": result.reason[:100],
                }
            except Exception:
                return None

        def _ap_scan_stock(sym):
            if sym in held:
                return None
            try:
                snap = get_stock_snapshot(self.alpaca, sym)
                result = strategy.analyze(snap)
                return {
                    "symbol": sym, "price": snap["price"], "snap": snap,
                    "result": result, "signal": result.signal.name,
                    "confidence": result.confidence,
                    "whale": 0.0, "sentiment": 0.0,
                    "reason": result.reason[:100],
                }
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit crypto
            futures = [executor.submit(_ap_scan_crypto, sym) for sym in TICKER_SYMBOLS]
            # Submit stocks if connected
            if self.alpaca:
                futures += [executor.submit(_ap_scan_stock, sym) for sym in STOCK_SYMBOLS]

            for future in as_completed(futures):
                r = future.result()
                if not r:
                    continue
                scanned += 1
                prices[r["symbol"]] = r["price"]
                scan_details.append({
                    "symbol": r["symbol"], "price": round(r["price"], 2),
                    "signal": r["signal"],
                    "confidence": round(r["confidence"], 3),
                    "whale": round(r["whale"], 2),
                    "sentiment": round(r["sentiment"], 2),
                    "reason": r["reason"],
                })
                if r["result"].signal == Signal.BUY and r["confidence"] >= min_entry_conf:
                    candidates.append((r["confidence"], r["symbol"], r["snap"], r["result"]))

        # Save per-symbol decision reasoning for the autopilot scan
        for detail in scan_details:
            if detail["signal"] == "BUY" and detail["confidence"] >= min_entry_conf:
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

        # Deep research on top candidates before opening (max 3 to control costs)
        from vesper.ai_research import research_asset
        deep_research_map = {}
        for _conf, sym, snap, _result in to_open[:3]:
            try:
                asset_type = "stock" if is_stock_symbol(sym) else "crypto"
                research = research_asset(
                    sym, self._snapshot_indicators(snap),
                    snap["price"], asset_type=asset_type,
                )
                deep_research_map[sym] = research
            except Exception:
                deep_research_map[sym] = {}

        # Allocate funds evenly across new positions
        per_position = available / len(to_open)
        actions = []

        for confidence, sym, snap, result in to_open:
            new_price = snap["price"]
            amount_usd = min(per_position, available)
            if amount_usd < 10:
                break

            deep = deep_research_map.get(sym, {})

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

            # Build rich strategy reason with deep search
            reason_parts = [f"Autopilot AI: {result.reason[:150]}"]
            if deep.get("researched"):
                reason_parts.append(f"Deep Search ({deep.get('engine', 'AI')}): {deep.get('reasoning', '')[:200]}")
                if deep.get("bullish_factors"):
                    reason_parts.append("Bullish: " + "; ".join(deep["bullish_factors"][:3]))
                if deep.get("bearish_factors"):
                    reason_parts.append("Bearish: " + "; ".join(deep["bearish_factors"][:3]))

            position = Position(
                symbol=sym, side="buy",
                entry_price=new_price,
                amount=limits.position_size_asset,
                cost_usd=amount_usd,
                limits=limits,
                strategy_reason=" | ".join(reason_parts),
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
                # Store full analysis for "Learn More"
                analysis_data = {
                    "indicators": self._snapshot_indicators(snap),
                    "strategy_signals": result.reason,
                    "confidence": round(confidence, 3),
                    "whale_score": round(snap.get("whale_score", 0), 2),
                    "sentiment_score": round(snap.get("sentiment_score", 0), 2),
                    "fear_greed": snap.get("fear_greed"),
                    "deep_research": {
                        "signal": deep.get("signal", ""),
                        "confidence": deep.get("confidence", 0),
                        "reasoning": deep.get("reasoning", ""),
                        "bullish_factors": deep.get("bullish_factors", []),
                        "bearish_factors": deep.get("bearish_factors", []),
                        "catalysts": deep.get("catalysts", []),
                        "sources": deep.get("sources", []),
                        "search_summary": deep.get("search_summary", ""),
                        "engine": deep.get("engine", ""),
                    },
                    "risk_level": risk_level,
                }
                self.portfolio._save_position_analysis(position.id, analysis_data)

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


# Shared state (separate module avoids __main__ vs vesper.main split)
from vesper.state import cycle_state as _cycle_state


class Vesper:
    """Main orchestrator — manages all user bots."""

    def __init__(self):
        self.logger = setup_logger()

    def run_all_users(self):
        """Run a trading cycle for every active user."""
        import time as _time
        _cycle_state["total_cycles"] += 1
        _cycle_state["last_cycle_time"] = _time.time()

        from vesper.dashboard.database import get_active_users, init_db
        init_db()

        active_users = get_active_users()
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Trading cycle #{_cycle_state['total_cycles']} — {len(active_users)} active user(s)")

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
                _cycle_state["exchange_name"] = bot._exchange_name or ""
                _cycle_state["last_cycle_error"] = ""
            except Exception as e:
                _cycle_state["last_cycle_error"] = f"{user.email}: {str(e)[:200]}"
                self.logger.error(f"[User:{user.email}] Cycle failed: {e}")

    def start(self):
        _cycle_state["startup_phase"] = "start_called"
        console.print(f"\n[bold magenta]{'='*60}[/]")
        console.print(f"[bold magenta]  VESPER — Crypto Trading Platform v0.2.0[/]")
        console.print(f"[bold magenta]{'='*60}[/]\n")

        # Start dashboard
        _cycle_state["startup_phase"] = "starting_dashboard"
        self._start_dashboard()

        # Run first cycle
        _cycle_state["startup_phase"] = "first_cycle"
        self.run_all_users()
        _cycle_state["startup_phase"] = "scheduler_running"

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
