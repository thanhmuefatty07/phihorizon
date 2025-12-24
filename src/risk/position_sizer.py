#!/usr/bin/env python3
"""
PhiHorizon - Position Sizing and Risk Management

Research-backed position sizing for trading:
1. Fractional Kelly Criterion (conservative)
2. Volatility-based position sizing
3. ATR-based stop loss and take profit

Key insight: Even 50% accuracy can profit with proper risk management.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Risk management configuration."""
    # Kelly sizing
    kelly_fraction: float = 0.25       # Quarter Kelly (conservative)
    max_position_pct: float = 0.10     # Max 10% of account per trade
    min_position_pct: float = 0.01     # Min 1% (avoid dust trades)
    
    # Stop loss / Take profit
    atr_stop_multiplier: float = 2.0   # Stop at 2x ATR
    atr_tp_multiplier: float = 3.0     # TP at 3x ATR (1.5x risk)
    atr_period: int = 14               # ATR lookback
    
    # Risk per trade
    max_risk_pct: float = 0.02         # Risk max 2% per trade


class PositionSizer:
    """
    Calculate optimal position sizes based on Kelly criterion and volatility.
    
    Usage:
        sizer = PositionSizer(account_size=10000)
        position = sizer.calculate_position(win_rate=0.55, avg_win=0.02, avg_loss=0.01)
        stop_loss = sizer.calculate_stop_loss(atr=100)
    """
    
    def __init__(
        self,
        account_size: float = 10000,
        config: RiskConfig = None,
    ):
        self.account_size = account_size
        self.config = config or RiskConfig()
    
    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Calculate Kelly fraction for optimal betting.
        
        Kelly formula: f* = (p * b - q) / b
        where:
            p = probability of winning
            q = probability of losing (1 - p)
            b = win/loss ratio
        
        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average win amount (as decimal, e.g., 0.02 for 2%)
            avg_loss: Average loss amount (as decimal, positive)
            
        Returns:
            Optimal fraction of account to bet (0-1)
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        p = win_rate
        q = 1 - p
        b = abs(avg_win) / abs(avg_loss)
        
        # Full Kelly
        kelly = (p * b - q) / b
        
        # Apply fraction (quarter Kelly is conservative)
        kelly_adj = kelly * self.config.kelly_fraction
        
        # Bound to reasonable range
        kelly_adj = max(0, min(kelly_adj, self.config.max_position_pct))
        
        return kelly_adj
    
    def calculate_position_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        current_price: float = 1.0,
    ) -> Dict[str, float]:
        """
        Calculate position size in units and dollars.
        
        Args:
            win_rate: Historical win rate
            avg_win: Average win percentage
            avg_loss: Average loss percentage
            current_price: Current asset price
            
        Returns:
            Dict with position details
        """
        kelly = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        
        # Position in dollars
        position_dollars = self.account_size * kelly
        
        # Position in units
        position_units = position_dollars / current_price if current_price > 0 else 0
        
        return {
            'kelly_fraction': kelly,
            'position_dollars': position_dollars,
            'position_units': position_units,
            'position_pct': kelly,
            'account_size': self.account_size,
        }
    
    def calculate_volatility_position(
        self,
        volatility: float,
        target_risk_pct: float = None,
    ) -> float:
        """
        Calculate position size based on volatility targeting.
        
        Higher volatility → smaller position
        Lower volatility → larger position
        
        Args:
            volatility: Current volatility (std of returns)
            target_risk_pct: Target portfolio risk (default from config)
            
        Returns:
            Position size as fraction of account
        """
        target_risk = target_risk_pct or self.config.max_risk_pct
        
        if volatility <= 0:
            return self.config.min_position_pct
        
        # Position size inversely proportional to volatility
        position = target_risk / volatility
        
        # Bound to limits
        position = max(self.config.min_position_pct,
                      min(position, self.config.max_position_pct))
        
        return position
    
    def calculate_stop_loss(
        self,
        atr: float,
        entry_price: float = None,
        direction: str = 'long',
    ) -> Dict[str, float]:
        """
        Calculate ATR-based stop loss level.
        
        Args:
            atr: Average True Range value
            entry_price: Entry price (optional, for absolute levels)
            direction: 'long' or 'short'
            
        Returns:
            Dict with stop loss details
        """
        stop_distance = atr * self.config.atr_stop_multiplier
        
        result = {
            'stop_distance': stop_distance,
            'stop_distance_pct': stop_distance / entry_price if entry_price else None,
            'atr': atr,
            'multiplier': self.config.atr_stop_multiplier,
        }
        
        if entry_price:
            if direction == 'long':
                result['stop_price'] = entry_price - stop_distance
            else:
                result['stop_price'] = entry_price + stop_distance
        
        return result
    
    def calculate_take_profit(
        self,
        atr: float,
        entry_price: float = None,
        direction: str = 'long',
    ) -> Dict[str, float]:
        """
        Calculate ATR-based take profit level.
        
        TP is set at 1.5x the stop distance for positive expectancy.
        
        Args:
            atr: Average True Range value
            entry_price: Entry price
            direction: 'long' or 'short'
            
        Returns:
            Dict with take profit details
        """
        tp_distance = atr * self.config.atr_tp_multiplier
        
        result = {
            'tp_distance': tp_distance,
            'tp_distance_pct': tp_distance / entry_price if entry_price else None,
            'atr': atr,
            'multiplier': self.config.atr_tp_multiplier,
            'risk_reward': self.config.atr_tp_multiplier / self.config.atr_stop_multiplier,
        }
        
        if entry_price:
            if direction == 'long':
                result['tp_price'] = entry_price + tp_distance
            else:
                result['tp_price'] = entry_price - tp_distance
        
        return result
    
    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = None,
    ) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period (default from config)
            
        Returns:
            ATR series
        """
        period = period or self.config.atr_period
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def get_trade_plan(
        self,
        entry_price: float,
        atr: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        direction: str = 'long',
    ) -> Dict:
        """
        Generate complete trade plan with position, stops, and targets.
        
        Args:
            entry_price: Planned entry price
            atr: Current ATR
            win_rate: Historical win rate
            avg_win: Average win
            avg_loss: Average loss
            direction: Trade direction
            
        Returns:
            Complete trade plan dictionary
        """
        position = self.calculate_position_size(win_rate, avg_win, avg_loss, entry_price)
        stop = self.calculate_stop_loss(atr, entry_price, direction)
        take_profit = self.calculate_take_profit(atr, entry_price, direction)
        
        return {
            'entry_price': entry_price,
            'direction': direction,
            'position': position,
            'stop_loss': stop,
            'take_profit': take_profit,
            'risk_reward': take_profit['risk_reward'],
            'max_loss_dollars': position['position_dollars'] * (stop['stop_distance'] / entry_price),
        }


def calculate_required_win_rate(avg_win: float, avg_loss: float) -> float:
    """
    Calculate minimum win rate needed for profitability.
    
    Break-even win rate = 1 / (1 + win/loss ratio)
    
    Args:
        avg_win: Average winning trade
        avg_loss: Average losing trade
        
    Returns:
        Minimum win rate for profitability
    """
    if avg_win <= 0 or avg_loss <= 0:
        return 1.0
    
    ratio = avg_win / avg_loss
    return 1 / (1 + ratio)


def calculate_expectancy(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """
    Calculate trade expectancy (expected value per trade).
    
    Expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    Args:
        win_rate: Probability of winning
        avg_win: Average win amount
        avg_loss: Average loss amount
        
    Returns:
        Expected value per trade
    """
    return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)


__all__ = [
    'RiskConfig',
    'PositionSizer',
    'calculate_required_win_rate',
    'calculate_expectancy',
]
