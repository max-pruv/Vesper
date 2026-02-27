"""Strategy catalog — user-facing strategy definitions with metadata."""

from dataclasses import dataclass, asdict


@dataclass
class StrategyInfo:
    id: str
    name: str
    description: str
    timeframe: str
    risk_level: str  # "low", "medium", "high"
    min_return_pct: float
    max_return_pct: float
    default_stop_loss_pct: float
    auto_managed: bool
    icon: str
    default_mode: str  # "one_off" or "continuous"


STRATEGY_CATALOG: list[StrategyInfo] = [
    StrategyInfo(
        id="next_hour_bet",
        name="Next Hour Bet",
        description="Bet on price direction for the next hour. Uses MACD crossover + RSI + volume to predict short-term movement. Fast in, fast out.",
        timeframe="1h",
        risk_level="high",
        min_return_pct=0.5,
        max_return_pct=2.0,
        default_stop_loss_pct=1.0,
        auto_managed=True,
        icon="⚡",
        default_mode="one_off",
    ),
    StrategyInfo(
        id="set_and_forget",
        name="Set & Forget",
        description="You set stop-loss and take-profit levels. The bot monitors 24/7 and closes when targets hit. True hands-off trading.",
        timeframe="Custom",
        risk_level="medium",
        min_return_pct=0.0,
        max_return_pct=0.0,
        default_stop_loss_pct=2.0,
        auto_managed=False,
        icon="🎯",
        default_mode="one_off",
    ),
    StrategyInfo(
        id="trend_rider",
        name="Trend Rider",
        description="Follows EMA 12/26 crossover trends. Enters on confirmed uptrend, rides it until EMA reversal. Best in trending markets.",
        timeframe="4h – 24h",
        risk_level="medium",
        min_return_pct=2.0,
        max_return_pct=8.0,
        default_stop_loss_pct=2.0,
        auto_managed=True,
        icon="📈",
        default_mode="continuous",
    ),
    StrategyInfo(
        id="mean_revert",
        name="Mean Revert",
        description="Buys when RSI < 30 and price hits lower Bollinger Band. Sells when overbought. Profits from price bouncing back to the mean.",
        timeframe="1h – 4h",
        risk_level="medium",
        min_return_pct=1.0,
        max_return_pct=4.0,
        default_stop_loss_pct=1.5,
        auto_managed=True,
        icon="🔄",
        default_mode="continuous",
    ),
    StrategyInfo(
        id="scalper",
        name="Scalper",
        description="Very fast trades catching small price movements. Uses MACD histogram + volume spikes. High frequency, small gains per trade.",
        timeframe="15m – 1h",
        risk_level="high",
        min_return_pct=0.3,
        max_return_pct=1.0,
        default_stop_loss_pct=0.5,
        auto_managed=True,
        icon="🏎️",
        default_mode="continuous",
    ),
    StrategyInfo(
        id="dca",
        name="DCA Accumulate",
        description="Dollar-cost averaging — buys a fixed $ amount at regular intervals regardless of price. No stop-loss. Long-term accumulation.",
        timeframe="Daily / Weekly",
        risk_level="low",
        min_return_pct=0.0,
        max_return_pct=0.0,
        default_stop_loss_pct=0.0,
        auto_managed=True,
        icon="🪙",
        default_mode="continuous",
    ),
    StrategyInfo(
        id="grid",
        name="Grid Trading",
        description="Places buy orders below current price and sell orders above at regular intervals. Profits from price oscillating in a range.",
        timeframe="Ongoing",
        risk_level="medium",
        min_return_pct=1.0,
        max_return_pct=5.0,
        default_stop_loss_pct=3.0,
        auto_managed=True,
        icon="📊",
        default_mode="continuous",
    ),
    StrategyInfo(
        id="breakout",
        name="Breakout Hunter",
        description="Waits for price to break above resistance with high volume. Enters on confirmed breakout. Big gains potential, higher risk.",
        timeframe="1h – 4h",
        risk_level="high",
        min_return_pct=3.0,
        max_return_pct=10.0,
        default_stop_loss_pct=2.0,
        auto_managed=True,
        icon="🚀",
        default_mode="one_off",
    ),
    StrategyInfo(
        id="ensemble",
        name="Full Auto",
        description="All 3 strategies (momentum + trend + mean reversion) vote together. Only trades when they agree. Fully automated.",
        timeframe="Ongoing",
        risk_level="medium",
        min_return_pct=1.5,
        max_return_pct=5.0,
        default_stop_loss_pct=2.0,
        auto_managed=True,
        icon="🤖",
        default_mode="continuous",
    ),
    StrategyInfo(
        id="trend_scanner",
        name="Trend Scanner",
        description="Scans the top 20 cryptocurrencies for strong trends. Auto-opens positions on the best opportunities and closes when momentum fades. Fully automated multi-asset scanning.",
        timeframe="Ongoing",
        risk_level="medium",
        min_return_pct=1.0,
        max_return_pct=8.0,
        default_stop_loss_pct=2.0,
        auto_managed=True,
        icon="🔍",
        default_mode="continuous",
    ),
]


def get_strategy_catalog() -> list[dict]:
    """Return the strategy catalog as a list of dicts for the frontend."""
    return [asdict(s) for s in STRATEGY_CATALOG]


def get_strategy_by_id(strategy_id: str) -> StrategyInfo | None:
    """Look up a strategy by its ID."""
    for s in STRATEGY_CATALOG:
        if s.id == strategy_id:
            return s
    return None
