"""Logging with rich console output."""

import logging
import os
from datetime import datetime

from rich.console import Console
from rich.table import Table

console = Console()


def setup_logger(name: str = "vesper", log_dir: str = "data") -> logging.Logger:
    """Configure logger with file + console output."""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    # File handler
    log_file = os.path.join(log_dir, f"vesper_{datetime.now():%Y%m%d}.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    return logger


def print_trade_signal(result, snapshot):
    """Print a formatted trade signal."""
    console.print(f"\n[bold cyan]{'='*60}[/]")
    console.print(f"[bold]Signal: {result.signal.value}[/] | "
                  f"Confidence: {result.confidence:.2f} | "
                  f"Price: ${snapshot['price']:,.2f}")
    console.print(f"[dim]{result.reason}[/]")
    console.print(f"[bold cyan]{'='*60}[/]\n")


def print_portfolio_summary(summary: dict):
    """Print a formatted portfolio summary."""
    table = Table(title="Portfolio Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    pnl_style = "green" if summary["total_pnl_usd"] >= 0 else "red"

    table.add_row("Total Value", f"${summary['total_value']:,.2f}")
    table.add_row("Cash", f"${summary['cash']:,.2f}")
    table.add_row("P&L", f"[{pnl_style}]${summary['total_pnl_usd']:+,.2f} ({summary['total_pnl_pct']:+.2f}%)[/]")
    table.add_row("Open Positions", str(summary["open_positions"]))
    table.add_row("Total Trades", str(summary["total_trades"]))
    table.add_row("Win Rate", f"{summary['win_rate']:.1f}%")

    console.print(table)


def print_position_opened(pos, limits):
    """Print position opened details."""
    console.print(f"[bold green]POSITION OPENED[/] {pos.symbol} {pos.side.upper()}")
    console.print(f"  Entry: ${pos.entry_price:,.2f} | Amount: {pos.amount:.6f} | Cost: ${pos.cost_usd:,.2f}")
    console.print(f"  Stop-Loss: ${limits.stop_loss_price:,.2f} | "
                  f"TP Min: ${limits.take_profit_min_price:,.2f} | "
                  f"TP Max: ${limits.take_profit_max_price:,.2f}")


def print_position_closed(record):
    """Print position closed details."""
    pnl_style = "green" if record.pnl_usd >= 0 else "red"
    console.print(f"[bold {pnl_style}]POSITION CLOSED[/] {record.symbol}")
    console.print(f"  Entry: ${record.entry_price:,.2f} -> Exit: ${record.exit_price:,.2f}")
    console.print(f"  P&L: [{pnl_style}]${record.pnl_usd:+,.2f} ({record.pnl_pct:+.2f}%)[/]")
    console.print(f"  Reason: {record.reason}")
