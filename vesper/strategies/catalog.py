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
        id="scalper",
        name="Scalper",
        description="Fast MACD momentum trades on 15-minute candles. Catches quick price swings with volume confirmation. Tight stop-loss, small frequent gains.",
        timeframe="15m",
        risk_level="high",
        min_return_pct=0.3,
        max_return_pct=1.0,
        default_stop_loss_pct=0.5,
        auto_managed=True,
        icon="⚡",
        default_mode="continuous",
    ),
    StrategyInfo(
        id="trend_rider",
        name="Trend Rider",
        description="Follows EMA 12/26 crossover trends on 4-hour candles. Enters on confirmed uptrend, rides it until reversal. Best in trending markets.",
        timeframe="4h",
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
        description="Buys when RSI < 30 and price hits lower Bollinger Band on 1-hour candles. Profits from price bouncing back to the mean. Best in ranging markets.",
        timeframe="1h",
        risk_level="medium",
        min_return_pct=1.0,
        max_return_pct=4.0,
        default_stop_loss_pct=1.5,
        auto_managed=True,
        icon="🔄",
        default_mode="continuous",
    ),
    StrategyInfo(
        id="smart_auto",
        name="Smart Auto",
        description="Multi-timeframe ensemble: 3 strategies (trend + momentum + mean reversion) vote on 1h+4h candles. Boosted by order book pressure, Fear/Greed Index, and ADX trend strength. The smartest mode.",
        timeframe="1h + 4h",
        risk_level="medium",
        min_return_pct=1.5,
        max_return_pct=5.0,
        default_stop_loss_pct=2.0,
        auto_managed=True,
        icon="🧠",
        default_mode="continuous",
    ),
    StrategyInfo(
        id="trend_scanner",
        name="Trend Scanner",
        description="Scans the top 20 cryptocurrencies using multi-timeframe Smart Auto analysis. Auto-opens on the coin with the strongest signal. When it closes, rescans and may switch to a different coin.",
        timeframe="1h + 4h",
        risk_level="medium",
        min_return_pct=1.0,
        max_return_pct=8.0,
        default_stop_loss_pct=2.0,
        auto_managed=True,
        icon="🔍",
        default_mode="continuous",
    ),
    StrategyInfo(
        id="altcoin_hunter",
        name="Altcoin Hunter",
        description="Scans 50+ altcoins with 6-factor trend scoring (momentum, RSI, EMA alignment, volume surge, ADX strength, BTC relative strength). Auto-allocates to the strongest trends, dynamically rebalances, and uses trailing stops. Fully autonomous.",
        timeframe="1h + 4h",
        risk_level="medium",
        min_return_pct=2.0,
        max_return_pct=15.0,
        default_stop_loss_pct=2.5,
        auto_managed=True,
        icon="🎯",
        default_mode="continuous",
    ),
    StrategyInfo(
        id="set_and_forget",
        name="Set & Forget",
        description="No AI — you set stop-loss and take-profit levels, the bot monitors 24/7 and closes when targets hit. Pure risk management, no automated entry.",
        timeframe="Custom",
        risk_level="medium",
        min_return_pct=0.0,
        max_return_pct=0.0,
        default_stop_loss_pct=2.0,
        auto_managed=False,
        icon="🎯",
        default_mode="one_off",
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
