#!/usr/bin/env python3
"""
PhiHorizon - Phi Filter Module

Uses Phi (IIT-inspired proxy) as a TRADING FILTER, not a prediction feature.

Strategy Change (V5.3):
- OLD: Add Phi as feature to XGBoost → doesn't improve accuracy
- NEW: Use Phi to FILTER signals → only trade when market shows high integration

Research basis:
- Phi may not predict direction, but may indicate "tradeable" vs "noisy" market
- High Phi = market is integrated/coherent → signals more reliable
- Low Phi = market is fragmented/noisy → avoid trading
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import our Phi proxies
try:
    from src.consciousness.metrics import calculate_phi_proxy
except ImportError:
    calculate_phi_proxy = None

logger = logging.getLogger(__name__)


@dataclass
class PhiFilterConfig:
    """Configuration for Phi filter."""
    threshold: float = 0.5           # Trade only when Phi > threshold
    method: str = 'ensemble'         # Which Phi proxy to use
    lookback_window: int = 100       # Window for Phi calculation
    require_stable: bool = True      # Require Phi to be stable over N bars
    stable_periods: int = 5          # Number of bars Phi must exceed threshold


class PhiFilter:
    """
    Filter trading signals based on Phi (consciousness) proxy.
    
    Usage:
        filter = PhiFilter(threshold=0.5)
        filtered_signals = filter.apply(signals, price_data)
    
    The filter REMOVES signals when Phi is low (market is noisy).
    """
    
    def __init__(self, config: PhiFilterConfig = None):
        self.config = config or PhiFilterConfig()
        self._phi_cache: Dict[str, float] = {}
    
    def calculate_phi(self, data: pd.DataFrame) -> float:
        """
        Calculate Phi for current data window.
        
        Args:
            data: OHLCV DataFrame with at least 'close' column
            
        Returns:
            Phi value (0-1)
        """
        if calculate_phi_proxy is None:
            logger.warning("Phi proxy not available, returning neutral 0.5")
            return 0.5
        
        if len(data) < self.config.lookback_window:
            return 0.5
        
        try:
            result = calculate_phi_proxy(
                data.tail(self.config.lookback_window),
                method=self.config.method,
            )
            return result.get(f'{self.config.method}_phi', 
                            result.get('ensemble_phi', 0.5))
        except Exception as e:
            logger.warning(f"Phi calculation failed: {e}")
            return 0.5
    
    def should_trade(self, data: pd.DataFrame) -> bool:
        """
        Determine if current market conditions are favorable for trading.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            True if Phi > threshold (market is integrated/tradeable)
        """
        phi = self.calculate_phi(data)
        return phi > self.config.threshold
    
    def calculate_phi_series(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate rolling Phi for entire dataset.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Series of Phi values indexed like input
        """
        phi_values = []
        window = self.config.lookback_window
        
        for i in range(len(data)):
            if i < window:
                phi_values.append(0.5)
            else:
                window_data = data.iloc[i-window:i]
                phi = self.calculate_phi(window_data)
                phi_values.append(phi)
        
        return pd.Series(phi_values, index=data.index)
    
    def filter_signals(
        self,
        signals: np.ndarray,
        data: pd.DataFrame,
        return_phi: bool = False,
    ) -> np.ndarray:
        """
        Filter trading signals based on Phi.
        
        Signals are KEPT only when Phi > threshold.
        
        Args:
            signals: Array of 0/1 trading signals
            data: OHLCV DataFrame aligned with signals
            return_phi: If True, also return Phi series
            
        Returns:
            Filtered signals (same shape as input)
        """
        if len(signals) != len(data):
            raise ValueError("signals and data must have same length")
        
        signals = np.asarray(signals)
        phi_series = self.calculate_phi_series(data)
        
        # Create filter mask
        phi_mask = phi_series > self.config.threshold
        
        # Apply stability requirement if configured
        if self.config.require_stable:
            stable_mask = self._calculate_stable_mask(phi_series)
            phi_mask = phi_mask & stable_mask
        
        # Apply filter
        filtered = signals * phi_mask.values.astype(int)
        
        # Log statistics
        original_trades = signals.sum()
        filtered_trades = filtered.sum()
        filter_rate = (1 - filtered_trades / original_trades) if original_trades > 0 else 0
        
        logger.info(f"Phi filter: {original_trades} → {filtered_trades} signals "
                   f"({filter_rate:.1%} filtered out)")
        
        if return_phi:
            return filtered, phi_series
        return filtered
    
    def _calculate_stable_mask(self, phi_series: pd.Series) -> pd.Series:
        """
        Create mask for periods where Phi is stable above threshold.
        
        Returns True only if Phi > threshold for last N periods.
        """
        above_threshold = phi_series > self.config.threshold
        stable = above_threshold.rolling(self.config.stable_periods).sum()
        return stable >= self.config.stable_periods
    
    def get_filter_statistics(
        self,
        signals: np.ndarray,
        data: pd.DataFrame,
        returns: np.ndarray = None,
    ) -> Dict:
        """
        Calculate statistics comparing filtered vs unfiltered signals.
        
        Args:
            signals: Original trading signals
            data: OHLCV data
            returns: Optional returns array for Sharpe calculation
            
        Returns:
            Dictionary with filter statistics
        """
        filtered, phi_series = self.filter_signals(signals, data, return_phi=True)
        
        stats = {
            'original_trades': int(signals.sum()),
            'filtered_trades': int(filtered.sum()),
            'filter_rate': float(1 - filtered.sum() / (signals.sum() + 1e-9)),
            'avg_phi': float(phi_series.mean()),
            'phi_above_threshold_pct': float((phi_series > self.config.threshold).mean()),
        }
        
        if returns is not None:
            returns = np.asarray(returns)
            
            # Original performance
            orig_returns = returns * signals
            orig_sharpe = orig_returns.mean() / (orig_returns.std() + 1e-9) * np.sqrt(252*24)
            
            # Filtered performance
            filt_returns = returns * filtered
            filt_sharpe = filt_returns.mean() / (filt_returns.std() + 1e-9) * np.sqrt(252*24)
            
            stats['original_sharpe'] = float(orig_sharpe)
            stats['filtered_sharpe'] = float(filt_sharpe)
            stats['sharpe_improvement'] = float(filt_sharpe - orig_sharpe)
        
        return stats


def create_phi_filter(
    threshold: float = 0.5,
    method: str = 'ensemble',
    require_stable: bool = True,
) -> PhiFilter:
    """
    Convenience function to create PhiFilter with common settings.
    
    Args:
        threshold: Phi threshold (0-1)
        method: Phi calculation method
        require_stable: Require stable Phi above threshold
        
    Returns:
        Configured PhiFilter instance
    """
    config = PhiFilterConfig(
        threshold=threshold,
        method=method,
        require_stable=require_stable,
    )
    return PhiFilter(config)


__all__ = [
    'PhiFilter',
    'PhiFilterConfig',
    'create_phi_filter',
]
