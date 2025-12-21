"""
Backtesting Strategy for Supreme System V5.

Trading signals for BACKTESTING only (not real-time).
Uses historical data simulation.

Hardware: Designed for i5 8th gen + 4GB RAM (laptop)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class BacktestSignal:
    """Trading signal for backtesting."""

    signal_type: SignalType
    price: float
    timestamp: pd.Timestamp
    confidence: float
    reason: str


class BacktestingStrategy:
    """
    Trading strategy for backtesting only.

    Uses RSI + SMA crossover logic on historical data.
    No real-time constraints, no cooldowns needed.

    Designed for:
    - Historical data analysis
    - Strategy validation
    - Benchmark comparison
    """

    def __init__(
        self,
        rsi_period: int = 14,
        sma_period: int = 20,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
    ):
        """
        Initialize backtesting strategy.

        Args:
            rsi_period: RSI calculation period
            sma_period: SMA calculation period
            rsi_oversold: RSI threshold for oversold (BUY)
            rsi_overbought: RSI threshold for overbought (SELL)
        """
        self.rsi_period = rsi_period
        self.sma_period = sma_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

        logger.info(f"BacktestingStrategy initialized: RSI({rsi_period}), SMA({sma_period})")

    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index).

        Args:
            prices: Close prices series

        Returns:
            RSI values (0-100)
        """
        delta = prices.diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(window=self.rsi_period, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=1).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_sma(self, prices: pd.Series) -> pd.Series:
        """Calculate Simple Moving Average."""
        return prices.rolling(window=self.sma_period, min_periods=1).mean()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for entire historical dataset.

        Args:
            df: DataFrame with 'close' column (and optionally 'timestamp')

        Returns:
            DataFrame with added columns: rsi, sma, signal
        """
        result = df.copy()

        # Calculate indicators
        result["rsi"] = self.calculate_rsi(result["close"])
        result["sma"] = self.calculate_sma(result["close"])

        # Generate signals
        # BUY: RSI < oversold AND price > SMA
        # SELL: RSI > overbought AND price < SMA
        # HOLD: otherwise

        conditions = [
            (result["rsi"] < self.rsi_oversold) & (result["close"] > result["sma"]),
            (result["rsi"] > self.rsi_overbought) & (result["close"] < result["sma"]),
        ]
        choices = [SignalType.BUY.value, SignalType.SELL.value]

        result["signal"] = np.select(conditions, choices, default=SignalType.HOLD.value)

        # Count signals
        buy_count = (result["signal"] == "buy").sum()
        sell_count = (result["signal"] == "sell").sum()
        logger.info(
            f"Generated signals: {buy_count} BUY, {sell_count} SELL, {len(result) - buy_count - sell_count} HOLD"
        )

        return result

    def get_signal_at_index(self, df: pd.DataFrame, idx: int) -> BacktestSignal:
        """
        Get single signal at specific index.

        Args:
            df: DataFrame with signals (from generate_signals)
            idx: Row index

        Returns:
            BacktestSignal object
        """
        row = df.iloc[idx]

        signal_str = row.get("signal", "hold")
        signal_type = SignalType(signal_str)

        timestamp = row.get("timestamp", pd.Timestamp.now())
        if isinstance(timestamp, (int, float)):
            timestamp = pd.Timestamp(timestamp, unit="s")

        return BacktestSignal(
            signal_type=signal_type,
            price=row["close"],
            timestamp=timestamp,
            confidence=self._calculate_confidence(row),
            reason=self._get_reason(row, signal_type),
        )

    def _calculate_confidence(self, row: pd.Series) -> float:
        """Calculate signal confidence based on indicator strength."""
        rsi = row.get("rsi", 50)

        # Confidence based on RSI extremity
        if rsi < self.rsi_oversold:
            return min(1.0, (self.rsi_oversold - rsi) / self.rsi_oversold)
        elif rsi > self.rsi_overbought:
            return min(1.0, (rsi - self.rsi_overbought) / (100 - self.rsi_overbought))
        else:
            return 0.5

    def _get_reason(self, row: pd.Series, signal: SignalType) -> str:
        """Generate human-readable reason for signal."""
        rsi = row.get("rsi", 50)
        price = row.get("close", 0)
        sma = row.get("sma", 0)

        if signal == SignalType.BUY:
            return f"RSI oversold ({rsi:.1f}) + price above SMA20 ({price:.2f} > {sma:.2f})"
        elif signal == SignalType.SELL:
            return f"RSI overbought ({rsi:.1f}) + price below SMA20 ({price:.2f} < {sma:.2f})"
        else:
            return f"Neutral conditions (RSI: {rsi:.1f})"
