"""Base strategy interface."""

from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class StrategyResult:
    signal: Signal
    confidence: float  # 0.0 to 1.0
    reason: str
    strategy_name: str


class Strategy(ABC):
    """Base class for all trading strategies."""

    name: str = "base"

    @abstractmethod
    def analyze(self, snapshot: dict) -> StrategyResult:
        """Analyze market data and return a trading signal."""
        ...
