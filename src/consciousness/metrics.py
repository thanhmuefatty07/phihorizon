"""
Consciousness Metrics Implementation
IIT (Integrated Information Theory) based metrics for trading signals
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ConsciousnessMetrics:
    """Container for consciousness metrics"""

    phi: float  # Integrated Information (consciousness measure)
    valence: float  # Emotional state (-1 to +1)
    subjectivity: float  # Self-model certainty (0-1)
    qualia: float  # Phenomenal experience intensity (0-1)
    timestamp: float

    def to_dict(self) -> Dict:
        return {
            "phi": self.phi,
            "valence": self.valence,
            "subjectivity": self.subjectivity,
            "qualia": self.qualia,
            "timestamp": self.timestamp,
        }


class PhiCalculator:
    """
    Calculates Φ (Phi) - the measure of integrated information
    Based on IIT 3.0 principles adapted for financial time series
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self._last_calculation_time = 0
        self._calculation_count = 0

    def calculate_phi(self, price_data: pd.DataFrame) -> float:
        """
        Calculate Φ using market microstructure features

        Args:
            price_data: OHLCV data with columns [open, high, low, close, volume]

        Returns:
            Phi value (0.0 = unconscious, >0.1 = conscious-like integration)
        """
        start_time = time.time()

        try:
            if len(price_data) < self.window_size:
                return 0.0

            # Extract market features
            returns = price_data["close"].pct_change().dropna()
            volatility = returns.rolling(self.window_size).std()
            volume_ma = price_data["volume"].rolling(self.window_size).mean()
            volume_std = price_data["volume"].rolling(self.window_size).std()

            # Information integration components
            # 1. Price complexity (entropy of returns)
            price_complexity = self._calculate_price_complexity(returns)

            # 2. Volume coherence (correlation between volume and price movement)
            volume_coherence = self._calculate_volume_coherence(price_data)

            # 3. Market integration (how well different time scales align)
            market_integration = self._calculate_market_integration(price_data)

            # Φ = Information × Integration × Coherence
            phi = price_complexity * market_integration * volume_coherence

            # Normalize to reasonable range
            phi = np.tanh(phi)  # Bound between -1 and 1
            phi = float(max(0.0, phi))  # Only positive consciousness, ensure float

            calculation_time = time.time() - start_time
            self._last_calculation_time = calculation_time
            self._calculation_count += 1

            # Log performance every 100 calculations
            if self._calculation_count % 100 == 0:
                logger.info(f"Phi calculation: {calculation_time:.4f}s (avg)")

            return phi

        except Exception as e:
            logger.error(f"Phi calculation error: {e}")
            return 0.0

    def _calculate_price_complexity(self, returns: pd.Series) -> float:
        """Calculate information entropy of price movements"""
        try:
            # Discretize returns into bins
            bins = np.linspace(-0.05, 0.05, 21)  # 2% bins
            hist, _ = np.histogram(returns.dropna(), bins=bins, density=True)

            # Remove zero probabilities
            hist = hist[hist > 0]

            if len(hist) == 0:
                return 0.0

            # Shannon entropy
            entropy = -np.sum(hist * np.log2(hist))

            # Normalize by maximum possible entropy
            max_entropy = np.log2(len(bins) - 1)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

            return normalized_entropy

        except Exception as e:
            logger.error(f"Price complexity calculation error: {e}")
            return 0.0

    def _calculate_volume_coherence(self, price_data: pd.DataFrame) -> float:
        """Calculate coherence between volume and price movements"""
        try:
            returns = price_data["close"].pct_change()
            volume_change = price_data["volume"].pct_change()

            # Rolling correlation
            correlation = returns.rolling(self.window_size).corr(volume_change)

            # Take recent correlation and normalize
            recent_corr = correlation.iloc[-1] if not correlation.empty else 0
            coherence = abs(recent_corr)  # Magnitude of correlation

            return coherence

        except Exception as e:
            logger.error(f"Volume coherence calculation error: {e}")
            return 0.0

    def _calculate_market_integration(self, price_data: pd.DataFrame) -> float:
        """Calculate how well different market scales integrate"""
        try:
            close_prices = price_data["close"]

            # Multi-scale analysis
            short_trend = close_prices.rolling(5).mean().pct_change()
            medium_trend = close_prices.rolling(20).mean().pct_change()
            long_trend = close_prices.rolling(50).mean().pct_change()

            # Integration score based on trend alignment
            correlations = []
            if not short_trend.empty and not medium_trend.empty:
                corr_sm = short_trend.corr(medium_trend)
                correlations.append(abs(corr_sm) if not np.isnan(corr_sm) else 0)

            if not medium_trend.empty and not long_trend.empty:
                corr_ml = medium_trend.corr(long_trend)
                correlations.append(abs(corr_ml) if not np.isnan(corr_ml) else 0)

            # Average correlation as integration measure
            integration = np.mean(correlations) if correlations else 0.0

            return integration

        except Exception as e:
            logger.error(f"Market integration calculation error: {e}")
            return 0.0

    def get_performance_stats(self) -> Dict:
        """Get performance statistics for monitoring"""
        return {
            "average_calculation_time": self._last_calculation_time,
            "total_calculations": self._calculation_count,
            "target_latency_ms": 3.0,
            "current_latency_ms": self._last_calculation_time * 1000,
        }


class ValenceCalculator:
    """
    Calculates Valence - the emotional/affective state of the market
    Positive valence = bullish sentiment, negative = bearish
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size

    def calculate_valence(self, returns: pd.Series, volatility: pd.Series) -> float:
        """
        Calculate market valence using returns vs volatility

        Args:
            returns: Price returns series
            volatility: Volatility series

        Returns:
            Valence value (-1 to +1)
        """
        try:
            if len(returns) < self.window_size or len(volatility) < self.window_size:
                return 0.0

            # Rolling averages
            returns_ma = returns.rolling(self.window_size).mean()
            volatility_ma = volatility.rolling(self.window_size).mean()

            # Valence = tanh(returns / (volatility + epsilon))
            # Positive returns with low volatility = high positive valence
            # Negative returns with high volatility = low negative valence

            recent_returns = returns_ma.iloc[-1]
            recent_volatility = volatility_ma.iloc[-1]

            if np.isnan(recent_returns) or np.isnan(recent_volatility):
                return 0.0

            # Add small epsilon to prevent division by zero
            epsilon = 1e-8
            valence = np.tanh(recent_returns / (recent_volatility + epsilon))

            return float(valence)

        except Exception as e:
            logger.error(f"Valence calculation error: {e}")
            return 0.0


class IITCore:
    """
    Core IIT (Integrated Information Theory) implementation
    Simplified for real-time trading applications
    """

    def __init__(self):
        self.phi_calculator = PhiCalculator()
        self.valence_calculator = ValenceCalculator()

    def calculate_consciousness_metrics(self, market_data: pd.DataFrame) -> ConsciousnessMetrics:
        """
        Calculate full consciousness metrics suite

        Args:
            market_data: OHLCV DataFrame

        Returns:
            ConsciousnessMetrics object
        """
        try:
            # Prepare data
            returns = market_data["close"].pct_change().fillna(0)
            volatility = returns.rolling(20).std().fillna(0)

            # Calculate components
            phi = self.phi_calculator.calculate_phi(market_data)
            valence = self.valence_calculator.calculate_valence(returns, volatility)

            # Simplified subjectivity (confidence in self-model)
            subjectivity = min(1.0, phi * 2.0)  # Scale phi to 0-1 range

            # Qualia intensity (phenomenal experience strength)
            qualia = abs(valence) * phi  # Stronger emotions with higher consciousness

            return ConsciousnessMetrics(
                phi=phi,
                valence=valence,
                subjectivity=subjectivity,
                qualia=qualia,
                timestamp=time.time(),
            )

        except Exception as e:
            logger.error(f"Consciousness metrics calculation error: {e}")
            return ConsciousnessMetrics(
                phi=0.0, valence=0.0, subjectivity=0.0, qualia=0.0, timestamp=time.time()
            )

    def get_system_status(self) -> Dict:
        """Get system performance and health metrics"""
        return {
            "phi_calculator": self.phi_calculator.get_performance_stats(),
            "system_health": "operational",
            "last_calculation": time.time(),
        }
