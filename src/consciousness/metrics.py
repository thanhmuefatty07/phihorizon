#!/usr/bin/env python3
"""
PhiHorizon - Consciousness Metrics (IIT-Inspired)

Implements Integrated Information Theory (IIT) inspired metrics for trading.

Classes:
    - PhiCalculator: Main calculator for Phi values
    - IITCore: Core IIT computations
    - MarketIntegrationMetrics: Container for market integration results

The Phi metric measures the degree of "integration" in market data:
    - High Phi → Market is coherent, patterns are meaningful
    - Low Phi → Market is noisy/random, patterns may be spurious

This uses simplified proxies (entropy-based) rather than full IIT Phi
(which would be computationally intractable for real-time trading).

References:
    - Tononi, G. (2004). "An information integration theory of consciousness"
    - Balduzzi, D. & Tononi, G. (2008). "Integrated Information in Discrete 
      Dynamical Systems: Motivation and Theoretical Framework"
    - IIT 4.0: https://integratedinformationtheory.org/
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MarketIntegrationMetrics:
    """Container for market integration analysis results."""
    
    phi: float = 0.0                    # Main Phi value (0-1)
    phi_normalized: float = 0.0         # Normalized Phi
    complexity: float = 0.0             # Market complexity
    integration: float = 0.0            # Integration measure
    entropy: float = 0.0                # Information entropy
    mutual_info: float = 0.0            # Mutual information between series
    transfer_entropy: float = 0.0       # Directional information flow
    
    # Derived indicators
    is_integrated: bool = False         # True if Phi > threshold
    regime: str = "unknown"             # Market regime classification
    confidence: float = 0.0             # Confidence in the measurement
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'phi': self.phi,
            'phi_normalized': self.phi_normalized,
            'complexity': self.complexity,
            'integration': self.integration,
            'entropy': self.entropy,
            'mutual_info': self.mutual_info,
            'transfer_entropy': self.transfer_entropy,
            'is_integrated': self.is_integrated,
            'regime': self.regime,
            'confidence': self.confidence,
        }


# =============================================================================
# CORE IIT COMPUTATIONS
# =============================================================================

class IITCore:
    """
    Core IIT (Integrated Information Theory) computations.
    
    This is a simplified implementation suitable for trading applications.
    Full IIT Phi computation (using pyphi) would be too slow for real-time use.
    
    Our approach:
    1. Discretize price returns into states
    2. Build transition probability matrix (TPM)
    3. Calculate information-theoretic measures:
       - Entropy (H)
       - Mutual Information (I)
       - Integration (Φ)
    
    Key insight: Phi = Whole_info - Sum(Parts_info)
    High Phi means the whole is greater than sum of parts (integrated system)
    """
    
    def __init__(
        self,
        n_states: int = 4,
        discretize_method: str = 'quantile',
    ):
        """
        Initialize IIT Core.
        
        Args:
            n_states: Number of discrete states (4 = quartiles)
            discretize_method: How to discretize continuous data
        """
        self.n_states = n_states
        self.discretize_method = discretize_method
    
    def discretize(self, data: np.ndarray) -> np.ndarray:
        """
        Discretize continuous data into discrete states.
        
        Args:
            data: Continuous values (e.g., returns)
            
        Returns:
            Integer state indices (0 to n_states-1)
        """
        if len(data) < self.n_states:
            return np.zeros(len(data), dtype=int)
        
        if self.discretize_method == 'quantile':
            # Use quantile-based discretization
            try:
                quantiles = np.linspace(0, 100, self.n_states + 1)
                bins = np.percentile(data, quantiles)
                # Handle edge case where all values are same
                bins = np.unique(bins)
                if len(bins) < 2:
                    return np.zeros(len(data), dtype=int)
                states = np.digitize(data, bins[1:-1])
            except Exception:
                states = np.zeros(len(data), dtype=int)
        else:
            # Equal-width bins
            data_min, data_max = data.min(), data.max()
            if data_max - data_min < 1e-10:
                return np.zeros(len(data), dtype=int)
            bins = np.linspace(data_min, data_max, self.n_states + 1)
            states = np.digitize(data, bins[1:-1])
        
        return np.clip(states, 0, self.n_states - 1)
    
    def build_tpm(self, states: np.ndarray) -> np.ndarray:
        """
        Build Transition Probability Matrix from state sequence.
        
        Args:
            states: Sequence of discrete states
            
        Returns:
            TPM of shape (n_states, n_states)
        """
        tpm = np.zeros((self.n_states, self.n_states))
        
        for i in range(len(states) - 1):
            current = int(states[i])
            next_state = int(states[i + 1])
            if 0 <= current < self.n_states and 0 <= next_state < self.n_states:
                tpm[current, next_state] += 1
        
        # Normalize rows
        row_sums = tpm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        tpm = tpm / row_sums
        
        return tpm
    
    def entropy(self, probs: np.ndarray) -> float:
        """
        Calculate Shannon entropy.
        
        H(X) = -Σ p(x) log₂ p(x)
        
        Args:
            probs: Probability distribution
            
        Returns:
            Entropy in bits
        """
        probs = np.asarray(probs).flatten()
        probs = probs[probs > 0]  # Remove zeros
        
        if len(probs) == 0:
            return 0.0
        
        return -np.sum(probs * np.log2(probs))
    
    def mutual_information(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Calculate mutual information between two series.
        
        I(X;Y) = H(X) + H(Y) - H(X,Y)
        
        Args:
            x, y: Two time series (same length)
            
        Returns:
            Mutual information in bits
        """
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        # Discretize
        x_states = self.discretize(np.asarray(x))
        y_states = self.discretize(np.asarray(y))
        
        # Calculate marginal distributions
        x_probs = np.bincount(x_states, minlength=self.n_states) / len(x_states)
        y_probs = np.bincount(y_states, minlength=self.n_states) / len(y_states)
        
        # Calculate joint distribution
        joint = np.zeros((self.n_states, self.n_states))
        for i in range(len(x_states)):
            joint[x_states[i], y_states[i]] += 1
        joint /= len(x_states)
        
        # I(X;Y) = H(X) + H(Y) - H(X,Y)
        h_x = self.entropy(x_probs)
        h_y = self.entropy(y_probs)
        h_xy = self.entropy(joint)
        
        return max(0, h_x + h_y - h_xy)
    
    def calculate_phi(
        self,
        data: Union[np.ndarray, pd.DataFrame],
    ) -> float:
        """
        Calculate Phi (integrated information) proxy.
        
        Phi measures how much "more" information is in the whole system
        compared to its parts.
        
        For a trading application:
        - Split market data into components (returns, volume, etc.)
        - Calculate whole-system mutual information
        - Calculate part-by-part information
        - Phi = Whole - Sum(Parts)
        
        Args:
            data: Market data (returns or OHLCV)
            
        Returns:
            Phi value (0-1, normalized)
        """
        if isinstance(data, pd.DataFrame):
            # Use multiple columns if available
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna().values
            else:
                returns = data.iloc[:, 0].pct_change().dropna().values
            
            # Get volume if available
            if 'volume' in data.columns:
                volume = data['volume'].pct_change().dropna().values
                if len(volume) == len(returns):
                    return self._calculate_bivariate_phi(returns, volume)
        else:
            returns = np.asarray(data)
            if len(returns.shape) > 1 and returns.shape[1] >= 2:
                return self._calculate_bivariate_phi(returns[:, 0], returns[:, 1])
        
        # Univariate Phi proxy using autocorrelation
        return self._calculate_univariate_phi(returns)
    
    def _calculate_univariate_phi(self, data: np.ndarray) -> float:
        """
        Calculate Phi for univariate data.
        
        Uses self-information and complexity measures.
        """
        if len(data) < 10:
            return 0.5
        
        states = self.discretize(data)
        tpm = self.build_tpm(states)
        
        # Calculate entropy of transition matrix
        # High entropy = random → low Phi
        # Low entropy = deterministic → potentially high Phi
        # But pure determinism is also low Phi (no integration needed)
        # Sweet spot is in middle → complexity
        
        state_probs = np.bincount(states, minlength=self.n_states) / len(states)
        h_current = self.entropy(state_probs)
        
        # Average conditional entropy H(next|current)
        h_cond = 0
        for i, prob in enumerate(state_probs):
            if prob > 0:
                h_cond += prob * self.entropy(tpm[i])
        
        # Mutual information with self (lagged)
        mi_self = h_current - h_cond
        
        # Normalize to 0-1
        max_entropy = np.log2(self.n_states)
        if max_entropy > 0:
            phi = mi_self / max_entropy
        else:
            phi = 0.5
        
        return np.clip(phi, 0, 1)
    
    def _calculate_bivariate_phi(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Calculate Phi for bivariate data.
        
        Phi = I(X;Y) - (I_x + I_y) normalized
        Where I_x, I_y are self-information of subparts
        """
        if len(x) < 10 or len(y) < 10:
            return 0.5
        
        # Mutual information between series
        mi_xy = self.mutual_information(x, y)
        
        # Self-information of each
        phi_x = self._calculate_univariate_phi(x)
        phi_y = self._calculate_univariate_phi(y)
        
        # Combined Phi: integration beyond individual parts
        # Normalize by max possible MI
        max_mi = np.log2(self.n_states)
        
        if max_mi > 0:
            integration = mi_xy / max_mi
        else:
            integration = 0.5
        
        # Phi = integration adjusted by individual complexities
        # High integration + moderate individual complexity = high Phi
        phi = (integration + phi_x + phi_y) / 3
        
        return np.clip(phi, 0, 1)


# =============================================================================
# PHI CALCULATOR (Main interface)
# =============================================================================

class PhiCalculator:
    """
    Main interface for calculating Phi (consciousness/integration) metrics.
    
    Usage:
        calculator = PhiCalculator()
        result = calculator.calculate(price_data)
        
        if result.is_integrated:
            # Market is coherent, signals may be reliable
            execute_trade()
        else:
            # Market is noisy, skip trading
            pass
    
    The Phi metric is inspired by Integrated Information Theory (IIT),
    but uses simplified proxies suitable for real-time trading.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        lookback: int = 100,
        n_states: int = 4,
    ):
        """
        Initialize PhiCalculator.
        
        Args:
            threshold: Phi threshold for "integrated" classification
            lookback: Default lookback window
            n_states: Number of discrete states for IIT calculation
        """
        self.threshold = threshold
        self.lookback = lookback
        self.core = IITCore(n_states=n_states)
    
    def calculate(
        self,
        data: Union[np.ndarray, pd.DataFrame, pd.Series],
        method: str = 'ensemble',
    ) -> MarketIntegrationMetrics:
        """
        Calculate comprehensive integration metrics.
        
        Args:
            data: Market data (OHLCV DataFrame or returns array)
            method: Calculation method ('iit', 'entropy', 'complexity', 'ensemble')
            
        Returns:
            MarketIntegrationMetrics with all computed values
        """
        result = MarketIntegrationMetrics()
        
        try:
            # Convert to appropriate format
            if isinstance(data, pd.Series):
                data = data.to_frame()
            
            if isinstance(data, pd.DataFrame):
                if len(data) < 10:
                    result.confidence = 0.1
                    return result
                
                # Use last N rows
                data = data.tail(self.lookback)
                
                # Calculate returns
                if 'close' in data.columns:
                    returns = data['close'].pct_change().dropna().values
                else:
                    returns = data.iloc[:, 0].pct_change().dropna().values
            else:
                returns = np.asarray(data).flatten()
            
            if len(returns) < 10:
                result.confidence = 0.1
                return result
            
            # Calculate Phi based on method
            if method == 'iit':
                phi = self.core.calculate_phi(data if isinstance(data, pd.DataFrame) else returns)
            elif method == 'entropy':
                phi = self._calculate_entropy_phi(returns)
            elif method == 'complexity':
                phi = self._calculate_complexity_phi(returns)
            else:  # ensemble
                phi_iit = self.core.calculate_phi(data if isinstance(data, pd.DataFrame) else returns)
                phi_ent = self._calculate_entropy_phi(returns)
                phi_cx = self._calculate_complexity_phi(returns)
                phi = (phi_iit + phi_ent + phi_cx) / 3
            
            # Fill in results
            result.phi = phi
            result.phi_normalized = phi
            result.complexity = self._calculate_complexity_phi(returns)
            result.entropy = self._calculate_raw_entropy(returns)
            result.is_integrated = phi > self.threshold
            result.regime = self._classify_regime(returns, phi)
            result.confidence = self._calculate_confidence(len(returns), phi)
            
        except Exception as e:
            logger.warning(f"Phi calculation error: {e}")
            result.confidence = 0.0
        
        return result
    
    def _calculate_entropy_phi(self, returns: np.ndarray) -> float:
        """
        Calculate Phi proxy based on entropy normalization.
        
        Idea: Neither too random (high H) nor too deterministic (low H)
        indicates integration.
        """
        # Normalize entropy to complexity measure
        # Max complexity at middle entropy
        states = self.core.discretize(returns)
        probs = np.bincount(states, minlength=self.core.n_states) / len(states)
        h = self.core.entropy(probs)
        max_h = np.log2(self.core.n_states)
        
        # Parabolic function: max at h/max_h = 0.5
        h_norm = h / max_h if max_h > 0 else 0.5
        phi = 4 * h_norm * (1 - h_norm)  # Peaks at 0.5
        
        return np.clip(phi, 0, 1)
    
    def _calculate_complexity_phi(self, returns: np.ndarray) -> float:
        """
        Calculate Phi based on statistical complexity.
        
        Uses Lempel-Ziv complexity as proxy for integration.
        """
        # Simple complexity: coefficient of variation of autocorrelations
        if len(returns) < 20:
            return 0.5
        
        # Calculate autocorrelations
        autocorrs = []
        for lag in range(1, min(10, len(returns) // 2)):
            if lag < len(returns):
                corr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                if not np.isnan(corr):
                    autocorrs.append(abs(corr))
        
        if len(autocorrs) < 2:
            return 0.5
        
        # Complexity = how structured (but not monotonic) the autocorrelations are
        autocorrs = np.array(autocorrs)
        mean_ac = autocorrs.mean()
        std_ac = autocorrs.std()
        
        # High mean + low std = deterministic pattern → medium Phi
        # Low mean + low std = random → low Phi
        # Medium mean + medium std = complex → high Phi
        complexity = mean_ac * (1 + std_ac / (mean_ac + 1e-10))
        
        # Normalize
        return np.clip(complexity * 2, 0, 1)
    
    def _calculate_raw_entropy(self, returns: np.ndarray) -> float:
        """Calculate raw entropy in bits."""
        states = self.core.discretize(returns)
        probs = np.bincount(states, minlength=self.core.n_states) / len(states)
        return self.core.entropy(probs)
    
    def _classify_regime(self, returns: np.ndarray, phi: float) -> str:
        """Classify market regime based on Phi and returns characteristics."""
        volatility = np.std(returns)
        trend = np.mean(returns)
        
        if phi > 0.7:
            if trend > 0:
                return "trending_up"
            elif trend < 0:
                return "trending_down"
            else:
                return "integrated_neutral"
        elif phi > 0.4:
            return "complex"
        else:
            if volatility > np.percentile(np.abs(returns), 75):
                return "chaotic"
            else:
                return "random"
    
    def _calculate_confidence(self, n_samples: int, phi: float) -> float:
        """Calculate confidence in the phi measurement."""
        # More samples = higher confidence
        sample_conf = min(1.0, n_samples / 100)
        
        # Middle phi values = higher confidence
        # Extreme values (0 or 1) may indicate issues
        phi_conf = 1 - 2 * abs(phi - 0.5)
        
        return (sample_conf + phi_conf) / 2


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_phi_proxy(
    data: Union[np.ndarray, pd.DataFrame],
    method: str = 'ensemble',
    **kwargs,
) -> Dict:
    """
    Convenience function to calculate Phi proxy.
    
    Args:
        data: Market data
        method: Calculation method
        **kwargs: Additional arguments for PhiCalculator
        
    Returns:
        Dictionary with phi values
    """
    calculator = PhiCalculator(**kwargs)
    result = calculator.calculate(data, method=method)
    
    return {
        'phi': result.phi,
        'ensemble_phi': result.phi,
        'iit_phi': result.phi,
        'entropy_phi': result.entropy,
        'complexity': result.complexity,
        'is_integrated': result.is_integrated,
        'regime': result.regime,
        'confidence': result.confidence,
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'PhiCalculator',
    'IITCore',
    'MarketIntegrationMetrics',
    'calculate_phi_proxy',
]
