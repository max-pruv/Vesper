"""Vesper — Main orchestrator and scheduler."""

import sys
import os
import signal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apscheduler.schedulers.blocking import BlockingScheduler

from config.settings import TradingConfig
from vesper.exchange import create_exchange, fetch_ticker
from vesper.market_data import get_market_snapshot
from vesper.strategies import EnsembleStrategy, Signal
from vesper.risk import RiskManager
from vesper.portfolio import Portfolio, Position
from vesper.logger import (
    setup_logger,
    print_trade_signal,
    print_portfolio_summary,
    print_position_opened,
    print_position_closed,
    console,
)


class Vesper:
    """Main trading bot orchestrator."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = setup_logger()
        self.strategy = EnsembleStrategy()
        self.risk_manager = RiskManager(config.risk)
        self.portfolio = Portfolio(
            initial_balance=config.paper_balance,
            data_dir="data",
        )

        if config.mode == "live":
            self.exchange = create_exchange(config.exchange)
            self.logger.info("Vesper started in LIVE mode — real trades will be executed")
        else:
            # Paper mode: still connect to exchange for market data (read-only)
            self.exchange = create_exchange(config.exchange)
            self.logger.info(
                f"Vesper started in PAPER mode — balance: ${config.paper_balance:.2f}"
            )

    def run_cycle(self):
        """Run one trading cycle: analyze all symbols, manage positions."""
        self.logger.info("=" * 60)
        self.logger.info("Starting trading cycle")

        prices = {}

        for symbol in self.config.symbols:
            try:
                self._process_symbol(symbol, prices)
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")

        # Print portfolio summary
        summary = self.portfolio.summary(prices)
        print_portfolio_summary(summary)
        self.logger.info(
            f"Portfolio: ${summary['total_value']:,.2f} | "
            f"P&L: ${summary['total_pnl_usd']:+,.2f} ({summary['total_pnl_pct']:+.2f}%) | "
            f"Win rate: {summary['win_rate']:.1f}%"
        )

    def _process_symbol(self, symbol: str, prices: dict):
        """Process a single symbol: check positions, analyze, trade."""
        self.logger.info(f"Analyzing {symbol}...")

        snapshot = get_market_snapshot(self.exchange, symbol)
        prices[symbol] = snapshot["price"]

        # Check existing positions for stop-loss / take-profit
        existing_pos = self.portfolio.get_position_for_symbol(symbol)
        if existing_pos:
            self._check_position_limits(existing_pos, snapshot["price"])
            return  # Don't open new position while one is active for this symbol

        # Analyze with ensemble strategy
        result = self.strategy.analyze(snapshot)
        print_trade_signal(result, snapshot)

        if result.signal == Signal.HOLD:
            self.logger.info(f"{symbol}: HOLD — no action")
            return

        # Check if we can open a new position
        if not self.risk_manager.can_open_position(len(self.portfolio.positions)):
            self.logger.info(f"Max concurrent positions reached, skipping {symbol}")
            return

        # Only act on BUY signals for spot trading
        if result.signal == Signal.BUY:
            self._open_position(symbol, snapshot, result)
        elif result.signal == Signal.SELL and existing_pos:
            # Close position on sell signal
            self._close_position_by_signal(existing_pos, snapshot["price"], result.reason)

    def _open_position(self, symbol, snapshot, result):
        """Open a new position."""
        portfolio_value = self.portfolio.total_value({symbol: snapshot["price"]})
        limits = self.risk_manager.calculate_position(
            entry_price=snapshot["price"],
            portfolio_value=portfolio_value,
            atr=snapshot["atr"],
            confidence=result.confidence,
            side="buy",
        )

        position = Position(
            symbol=symbol,
            side="buy",
            entry_price=snapshot["price"],
            amount=limits.position_size_asset,
            cost_usd=limits.position_size_usd,
            limits=limits,
            strategy_reason=result.reason,
        )

        if self.config.mode == "paper":
            if self.portfolio.open_position(position):
                print_position_opened(position, limits)
                self.logger.info(
                    f"PAPER BUY {symbol}: {position.amount:.6f} @ ${position.entry_price:,.2f} "
                    f"(${position.cost_usd:,.2f})"
                )
            else:
                self.logger.warning(f"Insufficient funds for {symbol} position")
        else:
            # Live trading
            try:
                from vesper.exchange import place_market_buy

                order = place_market_buy(self.exchange, symbol, limits.position_size_asset)
                position.entry_price = float(order.get("average", snapshot["price"]))
                position.amount = float(order.get("filled", limits.position_size_asset))
                position.cost_usd = position.entry_price * position.amount
                self.portfolio.open_position(position)
                print_position_opened(position, limits)
                self.logger.info(f"LIVE BUY {symbol}: order {order['id']} filled")
            except Exception as e:
                self.logger.error(f"Failed to execute BUY for {symbol}: {e}")

    def _check_position_limits(self, position: Position, current_price: float):
        """Check if position hit stop-loss or take-profit."""
        should_close, reason = self.risk_manager.should_close_position(
            current_price=current_price,
            entry_price=position.entry_price,
            limits=position.limits,
            side=position.side,
        )

        if should_close:
            self._close_position(position, current_price, reason)

    def _close_position(self, position: Position, exit_price: float, reason: str):
        """Close a position."""
        if self.config.mode == "live":
            try:
                from vesper.exchange import place_market_sell

                place_market_sell(self.exchange, position.symbol, position.amount)
            except Exception as e:
                self.logger.error(f"Failed to close position {position.id}: {e}")
                return

        record = self.portfolio.close_position(position.id, exit_price, reason)
        if record:
            print_position_closed(record)
            self.logger.info(
                f"CLOSED {position.symbol}: P&L ${record.pnl_usd:+,.2f} ({record.pnl_pct:+.2f}%) — {reason}"
            )

    def _close_position_by_signal(
        self, position: Position, current_price: float, reason: str
    ):
        """Close position triggered by strategy signal."""
        self._close_position(position, current_price, f"Strategy signal: {reason}")

    def start(self):
        """Start the scheduler — runs every interval."""
        console.print(f"\n[bold magenta]{'='*60}[/]")
        console.print(f"[bold magenta]  VESPER — Crypto Trading Bot v0.1.0[/]")
        console.print(f"[bold magenta]{'='*60}[/]")
        console.print(f"  Mode: [bold]{self.config.mode.upper()}[/]")
        console.print(f"  Symbols: {', '.join(self.config.symbols)}")
        console.print(f"  Interval: {self.config.interval_minutes} minutes")
        console.print(f"  Balance: ${self.config.paper_balance:,.2f}")
        console.print(f"[bold magenta]{'='*60}[/]\n")

        # Run first cycle immediately
        self.run_cycle()

        # Schedule subsequent cycles
        scheduler = BlockingScheduler()
        scheduler.add_job(
            self.run_cycle,
            "interval",
            minutes=self.config.interval_minutes,
            id="trading_cycle",
        )

        def shutdown(signum, frame):
            self.logger.info("Shutting down Vesper...")
            scheduler.shutdown(wait=False)

        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)

        self.logger.info(
            f"Scheduler started — next cycle in {self.config.interval_minutes} minutes"
        )
        scheduler.start()


def main():
    config = TradingConfig.from_env()
    bot = Vesper(config)
    bot.start()


if __name__ == "__main__":
    main()
