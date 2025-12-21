"""

Core Data Types with Memory Optimization.

Centralized data classes with __slots__ and string interning for memory efficiency.
"""

import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(slots=True)
class MarketTick:
    """Optimized market tick data."""

    symbol: str
    price: float
    volume: float
    timestamp: float

    def __post_init__(self):
        # Memory optimization: Intern symbol string to save RAM
        # Only useful if many objects share the same symbol string
        object.__setattr__(self, "symbol", sys.intern(self.symbol))


@dataclass(slots=True)
class Order:
    """Optimized order data with string interning."""

    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: float
    status: str = "NEW"

    def __post_init__(self):
        # Memory optimization: Intern repeated strings
        object.__setattr__(self, "symbol", sys.intern(self.symbol))
        object.__setattr__(self, "side", sys.intern(self.side))
        object.__setattr__(self, "status", sys.intern(self.status))


@dataclass(slots=True)
class Trade:
    """Optimized trade data with string interning."""

    symbol: str
    side: str
    quantity: float
    price: float
    pnl: float
    timestamp: float
    order_id: Optional[str] = None

    def __post_init__(self):
        # Memory optimization: Intern repeated strings
        object.__setattr__(self, "symbol", sys.intern(self.symbol))
        object.__setattr__(self, "side", sys.intern(self.side))
        if self.order_id:
            object.__setattr__(self, "order_id", sys.intern(self.order_id))


@dataclass(slots=True)
class Position:
    """Optimized position data with string interning."""

    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: float

    def __post_init__(self):
        # Memory optimization: Intern repeated strings
        object.__setattr__(self, "symbol", sys.intern(self.symbol))
        object.__setattr__(self, "side", sys.intern(self.side))


@dataclass(slots=True)
class RiskMetrics:
    """Optimized risk metrics data."""

    symbol: str
    position_value: float
    max_drawdown: float
    var_95: float
    sharpe_ratio: float
    timestamp: float

    def __post_init__(self):
        # Memory optimization: Intern symbol string
        object.__setattr__(self, "symbol", sys.intern(self.symbol))


@dataclass(slots=True)
class PerformanceSnapshot:
    """Optimized performance snapshot with string interning."""

    strategy_name: str
    total_pnl: float
    win_rate: float
    total_trades: int
    sharpe_ratio: float
    max_drawdown: float
    timestamp: float

    def __post_init__(self):
        # Memory optimization: Intern strategy name
        object.__setattr__(self, "strategy_name", sys.intern(self.strategy_name))
