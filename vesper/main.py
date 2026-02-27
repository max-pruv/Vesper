"""Vesper — Main orchestrator and scheduler for multi-user trading."""

import sys
import os
import signal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading

from apscheduler.schedulers.blocking import BlockingScheduler
import uvicorn

from config.settings import ExchangeConfig, RiskConfig, TICKER_SYMBOLS
from vesper.exchange import create_exchange
from vesper.market_data import get_market_snapshot
from vesper.strategies import EnsembleStrategy, Signal
from vesper.risk import RiskManager, PositionLimits
from vesper.portfolio import Portfolio, Position
from vesper.logger import setup_logger, console


class UserBot:
    """Trading bot instance for a single user."""

    def __init__(self, user_id: int, email: str, exchange_cfg: ExchangeConfig,
                 risk_cfg: RiskConfig, symbols: list[str], mode: str,
                 paper_balance: float, logger):
        self.user_id = user_id
        self.email = email
        self.mode = mode
        self.symbols = symbols
        self.logger = logger
        self.strategy = EnsembleStrategy()
        self.risk_manager = RiskManager(risk_cfg)

        data_dir = os.environ.get("VESPER_DATA_DIR", "data")
        self.portfolio = Portfolio(
            initial_balance=paper_balance,
            data_dir=data_dir,
            filename=f"portfolio_{user_id}.json",
        )

        self.exchange = create_exchange(exchange_cfg)

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

        summary = self.portfolio.summary(prices)
        self.logger.info(
            f"[User:{self.email}] Portfolio: ${summary['total_value']:,.2f} | "
            f"P&L: ${summary['total_pnl_usd']:+,.2f} ({summary['total_pnl_pct']:+.2f}%)"
        )

    def _process_symbol(self, symbol: str, prices: dict):
        snapshot = get_market_snapshot(self.exchange, symbol)
        prices[symbol] = snapshot["price"]

        closed_continuous = None
        existing_pos = self.portfolio.get_position_for_symbol(symbol)
        if existing_pos:
            should_close, reason = self.risk_manager.should_close_position(
                current_price=snapshot["price"],
                entry_price=existing_pos.entry_price,
                limits=existing_pos.limits,
                side=existing_pos.side,
            )
            if should_close:
                # Remember continuous params before closing
                if existing_pos.bet_mode == "continuous":
                    closed_continuous = existing_pos
                self._close_position(existing_pos, snapshot["price"], reason)
                # Continuous mode: fall through to re-evaluate
                if closed_continuous is None:
                    return  # One-off: done
            else:
                return  # Position still open, nothing to do

        # Analyze market signal
        result = self.strategy.analyze(snapshot)

        if result.signal == Signal.HOLD:
            return

        if not self.risk_manager.can_open_position(len(self.portfolio.positions)):
            return

        if result.signal == Signal.BUY:
            # Continuous re-entry: use original position parameters
            if closed_continuous:
                self._reopen_continuous(closed_continuous, snapshot)
            else:
                self._open_position(symbol, snapshot, result)

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

        limits = PositionLimits(
            stop_loss_price=new_price * (1 - sl_pct / 100),
            take_profit_min_price=new_price * (1 + tp_min_pct / 100),
            take_profit_max_price=new_price * (1 + tp_max_pct / 100),
            position_size_usd=amount_usd,
            position_size_asset=amount_usd / new_price,
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

        # Scan all major symbols for the strongest BUY signal
        best_signal = None
        best_snapshot = None
        best_symbol = None

        for sym in TICKER_SYMBOLS:
            try:
                if sym in prices:
                    # Already processed this symbol, skip fetching again
                    continue
                snap = get_market_snapshot(self.exchange, sym)
                prices[sym] = snap["price"]
                result = self.strategy.analyze(snap)
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
                bot = UserBot(
                    user_id=user.id,
                    email=user.email,
                    exchange_cfg=exchange_cfg,
                    risk_cfg=risk_cfg,
                    symbols=symbols,
                    mode=user.trading_mode,
                    paper_balance=user.paper_balance,
                    logger=self.logger,
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
