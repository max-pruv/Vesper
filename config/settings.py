import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ExchangeConfig:
    api_key: str = ""
    api_secret: str = ""
    sandbox: bool = False

    @classmethod
    def from_env(cls) -> "ExchangeConfig":
        return cls(
            api_key=os.getenv("COINBASE_API_KEY", ""),
            api_secret=os.getenv("COINBASE_API_SECRET", ""),
        )


@dataclass
class RiskConfig:
    stop_loss_pct: float = 2.0
    take_profit_min_pct: float = 1.5
    take_profit_max_pct: float = 5.0
    max_position_pct: float = 30.0
    max_concurrent_positions: int = 2

    @classmethod
    def from_env(cls) -> "RiskConfig":
        return cls(
            stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "2.0")),
            take_profit_min_pct=float(os.getenv("TAKE_PROFIT_MIN_PCT", "1.5")),
            take_profit_max_pct=float(os.getenv("TAKE_PROFIT_MAX_PCT", "5.0")),
            max_position_pct=float(os.getenv("MAX_POSITION_PCT", "30")),
            max_concurrent_positions=int(os.getenv("MAX_CONCURRENT_POSITIONS", "2")),
        )


TICKER_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "BNB/USDT",
    "LINK/USDT", "DOT/USDT", "MATIC/USDT", "UNI/USDT",
    "ATOM/USDT", "LTC/USDT", "NEAR/USDT", "APT/USDT",
    "ARB/USDT", "OP/USDT", "FIL/USDT", "AAVE/USDT",
]

# Extended altcoin universe for the Altcoin Hunter scanner
# Includes top 50+ altcoins available on Coinbase
ALTCOIN_UNIVERSE = TICKER_SYMBOLS + [
    "SUI/USDT", "SEI/USDT", "TIA/USDT", "INJ/USDT",
    "FET/USDT", "RENDER/USDT", "GRT/USDT", "IMX/USDT",
    "STX/USDT", "ALGO/USDT", "HBAR/USDT", "VET/USDT",
    "ICP/USDT", "SAND/USDT", "MANA/USDT", "AXS/USDT",
    "CRV/USDT", "LDO/USDT", "MKR/USDT", "SNX/USDT",
    "COMP/USDT", "SUSHI/USDT", "1INCH/USDT", "ENS/USDT",
    "DYDX/USDT", "PEPE/USDT", "SHIB/USDT", "FLOKI/USDT",
    "WIF/USDT", "BONK/USDT", "JUP/USDT", "WLD/USDT",
    "PYTH/USDT", "JTO/USDT", "ONDO/USDT", "ENA/USDT",
    "PENDLE/USDT", "ETHFI/USDT", "W/USDT", "ZRO/USDT",
]

# Stock symbols for Alpaca autopilot scanning
STOCK_SYMBOLS = [
    "AAPL/USD", "MSFT/USD", "GOOGL/USD", "AMZN/USD",
    "NVDA/USD", "META/USD", "TSLA/USD", "AMD/USD",
    "NFLX/USD", "CRM/USD", "AVGO/USD", "ORCL/USD",
    "PLTR/USD", "COIN/USD", "SQ/USD", "SHOP/USD",
    "UBER/USD", "ABNB/USD", "SNOW/USD", "MSTR/USD",
]


def is_stock_symbol(symbol: str) -> bool:
    """Check if a symbol is a stock (ends with /USD) vs crypto (/USDT)."""
    return symbol.endswith("/USD")


@dataclass
class TradingConfig:
    mode: str = "paper"
    paper_balance: float = 500.0
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    interval_minutes: int = 60
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    @classmethod
    def from_env(cls) -> "TradingConfig":
        symbols_str = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT")
        return cls(
            mode=os.getenv("TRADING_MODE", "paper"),
            paper_balance=float(os.getenv("PAPER_BALANCE", "500")),
            symbols=[s.strip() for s in symbols_str.split(",")],
            interval_minutes=int(os.getenv("INTERVAL_MINUTES", "60")),
            exchange=ExchangeConfig.from_env(),
            risk=RiskConfig.from_env(),
        )
