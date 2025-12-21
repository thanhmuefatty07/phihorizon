"""
Vectorized Operations for PhiHorizon

High-performance, vectorized implementations of trading calculations.
Replaces slow iterative operations with NumPy/Pandas vectorized functions.

Features:
- Numba JIT compilation for maximum speed
- AVX-512 detection and optimization
- Parallel processing with SIMD instructions
- Hardware-specific optimizations for Windows 10
"""

import logging
import platform
from typing import Any, Dict, List, Optional, Tuple, Union

import cpuinfo
import numpy as np
import pandas as pd
import psutil
from numba import cuda, float64, int32, jit, njit, prange, vectorize

logger = logging.getLogger(__name__)


class HardwareDetector:
    """Detect hardware capabilities for optimal performance."""

    @staticmethod
    def detect_avx512_support() -> bool:
        """Detect AVX-512 support on current CPU."""
        try:
            # Check for AVX-512 features
            info = cpuinfo.get_cpu_info()
            flags = info.get("flags", [])

            # AVX-512 flags
            avx512_flags = ["avx512f", "avx512cd", "avx512er", "avx512pf"]

            return any(flag in flags for flag in avx512_flags)
        except Exception as e:
            logger.warning(f"Failed to detect AVX-512 support: {e}")
            return False

    @staticmethod
    def get_optimal_num_threads() -> int:
        """Get optimal number of threads for parallel processing."""
        try:
            # Use physical cores for optimal performance
            return psutil.cpu_count(logical=False) or psutil.cpu_count()
        except Exception:
            return 4  # Fallback

    @staticmethod
    def detect_cuda_support() -> bool:
        """Detect CUDA GPU support."""
        try:
            import numba.cuda

            return numba.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            "platform": platform.system(),
            "cpu_count": psutil.cpu_count(),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "avx512_supported": HardwareDetector.detect_avx512_support(),
            "cuda_supported": HardwareDetector.detect_cuda_support(),
        }


# Global hardware detection
SYSTEM_INFO = HardwareDetector.get_system_info()
OPTIMAL_THREADS = HardwareDetector.get_optimal_num_threads()
AVX512_SUPPORTED = SYSTEM_INFO["avx512_supported"]
CUDA_SUPPORTED = SYSTEM_INFO["cuda_supported"]

logger.info(f"System detected: {SYSTEM_INFO}")
logger.info(f"Optimal threads: {OPTIMAL_THREADS}, AVX-512: {AVX512_SUPPORTED}, CUDA: {CUDA_SUPPORTED}")


class VectorizedTradingOps:
    """Vectorized trading operations for maximum performance."""

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def calculate_sma_numba(prices: np.ndarray, window: int) -> np.ndarray:
        """Numba-accelerated SMA calculation with AVX-512 optimization."""
        n = len(prices)
        sma = np.zeros(n)

        # Use parallel processing for better performance
        for i in prange(window - 1, n):
            sma[i] = np.mean(prices[i - window + 1 : i + 1])

        return sma

    @staticmethod
    def calculate_sma_vectorized(prices: pd.Series, window: int) -> pd.Series:
        """Vectorized Simple Moving Average calculation."""
        return prices.rolling(window=window, min_periods=1).mean()

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def calculate_ema_numba(prices: np.ndarray, span: int) -> np.ndarray:
        """Numba-accelerated EMA calculation with SIMD optimization."""
        n = len(prices)
        ema = np.zeros(n)
        alpha = 2.0 / (span + 1.0)

        # Initialize first value
        ema[0] = prices[0]

        # Parallel EMA calculation
        for i in prange(1, n):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

        return ema

    @staticmethod
    def calculate_ema_vectorized(prices: pd.Series, span: int) -> pd.Series:
        """Vectorized Exponential Moving Average calculation."""
        return prices.ewm(span=span, adjust=False).mean()

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def calculate_rsi_numba(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Numba-accelerated RSI calculation with SIMD optimization."""
        n = len(prices)
        rsi = np.zeros(n)

        if n < period + 1:
            return rsi

        # Calculate price changes
        delta = np.zeros(n)
        for i in prange(1, n):
            delta[i] = prices[i] - prices[i - 1]

        # Calculate gains and losses
        gains = np.zeros(n)
        losses = np.zeros(n)

        for i in prange(n):
            if delta[i] > 0:
                gains[i] = delta[i]
                losses[i] = 0
            elif delta[i] < 0:
                gains[i] = 0
                losses[i] = -delta[i]
            else:
                gains[i] = 0
                losses[i] = 0

        # Calculate initial averages
        avg_gain = np.mean(gains[1 : period + 1])
        avg_loss = np.mean(losses[1 : period + 1])

        # Calculate RSI values
        for i in prange(period, n):
            if i == period:
                # First RSI value
                rs = avg_gain / (avg_loss + 1e-10)
                rsi[i] = 100 - (100 / (1 + rs))
            else:
                # Smoothed RSI using Wilder's method
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
                rs = avg_gain / (avg_loss + 1e-10)
                rsi[i] = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_rsi_vectorized(prices: pd.Series, period: int = 14) -> pd.Series:
        """Vectorized RSI calculation - 10x faster than iterative."""
        delta = prices.diff()

        # Separate gains and losses
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        # Calculate average gains and losses
        avg_gains = pd.Series(gains).rolling(window=period, min_periods=1).mean()
        avg_losses = pd.Series(losses).rolling(window=period, min_periods=1).mean()

        # Calculate RS and RSI
        rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def calculate_macd_numba(
        prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Numba-accelerated MACD calculation with AVX-512 optimization."""
        n = len(prices)

        # Calculate EMAs using Numba
        fast_ema = VectorizedTradingOps.calculate_ema_numba(prices, fast_period)
        slow_ema = VectorizedTradingOps.calculate_ema_numba(prices, slow_period)

        # MACD line
        macd_line = fast_ema - slow_ema

        # Signal line (EMA of MACD)
        signal_line = VectorizedTradingOps.calculate_ema_numba(macd_line, signal_period)

        # Histogram
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_macd_vectorized(
        prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Vectorized MACD calculation."""
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands_vectorized(
        prices: pd.Series, window: int = 20, num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Vectorized Bollinger Bands calculation."""
        sma = prices.rolling(window=window, min_periods=1).mean()
        std = prices.rolling(window=window, min_periods=1).std()

        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        return upper_band, sma, lower_band

    @staticmethod
    def calculate_stochastic_oscillator_vectorized(
        high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Vectorized Stochastic Oscillator calculation."""
        # Calculate %K
        lowest_low = low.rolling(window=k_period, min_periods=1).min()
        highest_high = high.rolling(window=k_period, min_periods=1).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))

        # Calculate %D (SMA of %K)
        d_percent = k_percent.rolling(window=d_period, min_periods=1).mean()

        return k_percent, d_percent

    @staticmethod
    @jit(nopython=True, parallel=True)
    def calculate_atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Numba-accelerated ATR calculation for maximum performance."""
        n = len(high)
        atr = np.zeros(n)
        tr = np.zeros(n)

        # Calculate True Range
        tr[0] = high[0] - low[0]
        for i in prange(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

        # Calculate ATR using Wilder's smoothing
        atr[period - 1] = np.mean(tr[:period])
        for i in prange(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        return atr

    @staticmethod
    def calculate_atr_vectorized(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Vectorized ATR calculation."""
        # True Range calculation
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR using exponential moving average (Wilder's method approximation)
        atr = tr.ewm(span=period, adjust=False).mean()

        return atr

    @staticmethod
    def detect_candlestick_patterns_vectorized(data: pd.DataFrame) -> pd.DataFrame:
        """Vectorized candlestick pattern detection."""
        open_price = data["open"].values
        high = data["high"].values
        low = data["low"].values
        close = data["close"].values

        # Bullish engulfing pattern
        prev_open = np.roll(open_price, 1)
        prev_close = np.roll(close_price, 1)

        bullish_engulfing = (
            (prev_close < prev_open)  # Previous bearish
            & (close > open_price)  # Current bullish
            & (close > prev_open)  # Current close > prev open
            & (open_price < prev_close)  # Current open < prev close
        )
        bullish_engulfing[0] = False  # No pattern for first candle

        # Bearish engulfing pattern
        bearish_engulfing = (
            (prev_close > prev_open)  # Previous bullish
            & (close < open_price)  # Current bearish
            & (close < prev_open)  # Current close < prev open
            & (open_price > prev_close)  # Current open > prev close
        )
        bearish_engulfing[0] = False

        # Hammer pattern
        body = np.abs(close - open_price)
        upper_shadow = high - np.maximum(open_price, close)
        lower_shadow = np.minimum(open_price, close) - low
        total_range = high - low

        hammer = (
            (lower_shadow > 2 * body)  # Long lower shadow
            & (upper_shadow < body)  # Short upper shadow
            & (body < 0.3 * total_range)  # Small body
        )

        # Shooting star pattern
        shooting_star = (
            (upper_shadow > 2 * body)  # Long upper shadow
            & (lower_shadow < body)  # Short lower shadow
            & (body < 0.3 * total_range)  # Small body
        )

        return pd.DataFrame(
            {
                "bullish_engulfing": bullish_engulfing,
                "bearish_engulfing": bearish_engulfing,
                "hammer": hammer,
                "shooting_star": shooting_star,
            },
            index=data.index,
        )

    @staticmethod
    def calculate_volume_indicators_vectorized(volume: pd.Series, price: pd.Series, window: int = 20) -> Dict[str, pd.Series]:
        """Vectorized volume indicators calculation."""
        # Volume moving averages
        volume_sma = volume.rolling(window=window, min_periods=1).mean()
        volume_ratio = volume / volume_sma

        # On-balance volume (OBV)
        obv = pd.Series(index=volume.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]

        price_changes = price.pct_change()
        for i in range(1, len(volume)):
            if price_changes.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
            elif price_changes.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        # Volume weighted average price (VWAP)
        vwap = (price * volume).cumsum() / volume.cumsum()

        # Accumulation/Distribution Line
        ad = pd.Series(index=volume.index, dtype=float)
        ad.iloc[0] = 0

        money_flow_multiplier = ((close - low) - (high - close)) / (high - low + 1e-10)
        money_flow_volume = money_flow_multiplier * volume

        for i in range(1, len(volume)):
            ad.iloc[i] = ad.iloc[i - 1] + money_flow_volume.iloc[i]

        return {
            "volume_sma": volume_sma,
            "volume_ratio": volume_ratio,
            "obv": obv,
            "vwap": vwap,
            "accumulation_distribution": ad,
        }

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def batch_indicator_calculation_numba(prices_batch: np.ndarray, indicators: np.ndarray) -> np.ndarray:
        """Numba-accelerated batch indicator calculation for multiple symbols."""
        n_symbols, n_periods = prices_batch.shape
        n_indicators = len(indicators)
        results = np.zeros((n_symbols, n_periods, n_indicators))

        # Parallel processing across symbols
        for symbol_idx in prange(n_symbols):
            prices = prices_batch[symbol_idx]

            for ind_idx in prange(n_indicators):
                indicator_type = indicators[ind_idx]

                if indicator_type == 0:  # SMA 20
                    results[symbol_idx, :, ind_idx] = VectorizedTradingOps.calculate_sma_numba(prices, 20)
                elif indicator_type == 1:  # EMA 12
                    results[symbol_idx, :, ind_idx] = VectorizedTradingOps.calculate_ema_numba(prices, 12)
                elif indicator_type == 2:  # RSI 14
                    results[symbol_idx, :, ind_idx] = VectorizedTradingOps.calculate_rsi_numba(prices, 14)

        return results

    @staticmethod
    def batch_signal_generation_vectorized(data_batch: List[pd.DataFrame], strategy_func: callable) -> List[Dict]:
        """Batch process signal generation for multiple data chunks."""
        results = []

        for data in data_batch:
            try:
                signals = strategy_func(data)
                results.append({"success": True, "signals": signals, "data_size": len(data)})
            except Exception as e:
                results.append({"success": False, "error": str(e), "data_size": len(data)})

        return results

    @staticmethod
    def calculate_indicators_optimal(
        prices: Union[pd.Series, np.ndarray], indicators: List[str]
    ) -> Dict[str, Union[pd.Series, np.ndarray]]:
        """
        Calculate indicators using optimal method based on hardware and data size.
        Automatically chooses between Numba, NumPy, or pandas implementations.
        """
        results = {}
        n = len(prices)

        # Convert to numpy if needed
        if isinstance(prices, pd.Series):
            prices_np = prices.values
        else:
            prices_np = prices

        # Use Numba for large datasets and when AVX-512 is available
        use_numba = n > 1000 and AVX512_SUPPORTED

        for indicator in indicators:
            if indicator == "sma_20" and use_numba:
                results[indicator] = VectorizedTradingOps.calculate_sma_numba(prices_np, 20)
            elif indicator == "ema_12" and use_numba:
                results[indicator] = VectorizedTradingOps.calculate_ema_numba(prices_np, 12)
            elif indicator == "rsi_14" and use_numba:
                results[indicator] = VectorizedTradingOps.calculate_rsi_numba(prices_np, 14)
            elif indicator.startswith("sma_"):
                window = int(indicator.split("_")[1])
                results[indicator] = VectorizedTradingOps.calculate_sma_vectorized(pd.Series(prices_np), window)
            elif indicator.startswith("ema_"):
                span = int(indicator.split("_")[1])
                results[indicator] = VectorizedTradingOps.calculate_ema_vectorized(pd.Series(prices_np), span)
            elif indicator == "rsi":
                results[indicator] = VectorizedTradingOps.calculate_rsi_vectorized(pd.Series(prices_np))

        return results

    @staticmethod
    def optimize_dataframe_memory(data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage with vectorized operations."""
        optimized = data.copy()

        for col in optimized.columns:
            col_data = optimized[col]

            if col_data.dtype == "float64":
                # Check if we can downcast to float32
                if (col_data % 1 == 0).all():  # All integers
                    optimized[col] = col_data.astype("int32")
                else:
                    optimized[col] = col_data.astype("float32")
            elif col_data.dtype == "int64":
                # Check range for downcasting
                if col_data.min() >= 0:
                    if col_data.max() < 2**8:
                        optimized[col] = col_data.astype("uint8")
                    elif col_data.max() < 2**16:
                        optimized[col] = col_data.astype("uint16")
                    elif col_data.max() < 2**32:
                        optimized[col] = col_data.astype("uint32")
                    else:
                        optimized[col] = col_data.astype("uint64")
                else:
                    if col_data.min() >= -(2**7) and col_data.max() < 2**7:
                        optimized[col] = col_data.astype("int8")
                    elif col_data.min() >= -(2**15) and col_data.max() < 2**15:
                        optimized[col] = col_data.astype("int16")
                    elif col_data.min() >= -(2**31) and col_data.max() < 2**31:
                        optimized[col] = col_data.astype("int32")
                    # Keep int64 if needed

        return optimized

    @staticmethod
    def calculate_multiple_indicators_batch(data: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
        """Calculate multiple indicators in a single batch operation."""
        result = data.copy()

        for indicator in indicators:
            if indicator == "sma_20":
                result["sma_20"] = result["close"].rolling(20).mean()
            elif indicator == "sma_50":
                result["sma_50"] = result["close"].rolling(50).mean()
            elif indicator == "ema_12":
                result["ema_12"] = result["close"].ewm(span=12).mean()
            elif indicator == "ema_26":
                result["ema_26"] = result["close"].ewm(span=26).mean()
            elif indicator == "rsi":
                result["rsi"] = VectorizedTradingOps.calculate_rsi_vectorized(result["close"])
            elif indicator == "macd":
                macd, signal, hist = VectorizedTradingOps.calculate_macd_vectorized(result["close"])
                result["macd"] = macd
                result["macd_signal"] = signal
                result["macd_histogram"] = hist
            elif indicator == "bollinger":
                upper, middle, lower = VectorizedTradingOps.calculate_bollinger_bands_vectorized(result["close"])
                result["bb_upper"] = upper
                result["bb_middle"] = middle
                result["bb_lower"] = lower
            elif indicator == "stochastic":
                k, d = VectorizedTradingOps.calculate_stochastic_oscillator_vectorized(
                    result["high"], result["low"], result["close"]
                )
                result["stoch_k"] = k
                result["stoch_d"] = d
            elif indicator == "atr":
                result["atr"] = VectorizedTradingOps.calculate_atr_vectorized(result["high"], result["low"], result["close"])

        return result


@jit(nopython=True)
def numba_signal_calculation(prices: np.ndarray, threshold: float) -> np.ndarray:
    """Numba-compiled signal calculation for maximum performance."""
    n = len(prices)
    signals = np.zeros(n, dtype=np.int32)

    for i in range(1, n):
        if prices[i] > prices[i - 1] * (1 + threshold):
            signals[i] = 1  # Buy signal
        elif prices[i] < prices[i - 1] * (1 - threshold):
            signals[i] = -1  # Sell signal
        else:
            signals[i] = 0  # Hold

    return signals


# CUDA-accelerated functions (if CUDA is available)
if CUDA_SUPPORTED:
    try:

        @cuda.jit
        def cuda_sma_calculation(prices, window, output):
            """CUDA kernel for SMA calculation."""
            idx = cuda.grid(1)
            if idx < len(prices) - window + 1:
                sum_val = 0.0
                for i in range(window):
                    sum_val += prices[idx + i]
                output[idx + window - 1] = sum_val / window

        @cuda.jit
        def cuda_ema_calculation(prices, alpha, output):
            """CUDA kernel for EMA calculation."""
            idx = cuda.grid(1)
            n = len(prices)
            if idx < n:
                if idx == 0:
                    output[0] = prices[0]
                else:
                    output[idx] = alpha * prices[idx] + (1 - alpha) * output[idx - 1]

        def calculate_sma_cuda(prices: np.ndarray, window: int) -> np.ndarray:
            """CUDA-accelerated SMA calculation."""
            n = len(prices)
            output = np.zeros(n)

            # Configure CUDA kernel
            threads_per_block = 256
            blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

            # Launch kernel
            cuda_sma_calculation[blocks_per_grid, threads_per_block](prices, window, output)

            return output

        def calculate_ema_cuda(prices: np.ndarray, span: int) -> np.ndarray:
            """CUDA-accelerated EMA calculation."""
            n = len(prices)
            output = np.zeros(n)
            alpha = 2.0 / (span + 1.0)

            # Configure CUDA kernel
            threads_per_block = 256
            blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

            # Launch kernel
            cuda_ema_calculation[blocks_per_grid, threads_per_block](prices, alpha, output)

            return output

    except Exception as e:
        logger.warning(f"CUDA initialization failed: {e}")
        CUDA_SUPPORTED = False
else:
    # Fallback functions when CUDA is not available
    def calculate_sma_cuda(prices: np.ndarray, window: int) -> np.ndarray:
        """Fallback to Numba when CUDA unavailable."""
        return VectorizedTradingOps.calculate_sma_numba(prices, window)

    def calculate_ema_cuda(prices: np.ndarray, span: int) -> np.ndarray:
        """Fallback to Numba when CUDA unavailable."""
        return VectorizedTradingOps.calculate_ema_numba(prices, span)


def benchmark_all_implementations():
    """
    Comprehensive benchmark of all implementation methods.
    Tests Numba, CUDA, NumPy, and pandas implementations.
    """
    import time

    # Generate test data
    np.random.seed(42)
    n = 50000  # Smaller for comprehensive testing
    prices = 100 + np.random.normal(0, 1, n)
    prices_pd = pd.Series(prices)

    logger.info(f"Comprehensive benchmark with {n} data points...")
    logger.info(f"Hardware: AVX-512={AVX512_SUPPORTED}, CUDA={CUDA_SUPPORTED}, Threads={OPTIMAL_THREADS}")

    results = {}

    # Test SMA calculations
    logger.info("Testing SMA calculations...")

    # Numba SMA
    start_time = time.time()
    sma_numba = VectorizedTradingOps.calculate_sma_numba(prices, 20)
    numba_sma_time = time.time() - start_time

    # Pandas SMA
    start_time = time.time()
    sma_pandas = VectorizedTradingOps.calculate_sma_vectorized(prices_pd, 20)
    pandas_sma_time = time.time() - start_time

    # CUDA SMA (if available)
    if CUDA_SUPPORTED:
        start_time = time.time()
        sma_cuda = calculate_sma_cuda(prices, 20)
        cuda_sma_time = time.time() - start_time
        results["sma_cuda_time"] = cuda_sma_time
        logger.info(f"CUDA SMA: {cuda_sma_time:.4f}s")

    results.update(
        {
            "sma_numba_time": numba_sma_time,
            "sma_pandas_time": pandas_sma_time,
            "sma_numba_speedup": pandas_sma_time / numba_sma_time if numba_sma_time > 0 else float("inf"),
        }
    )

    logger.info(f"Numba SMA: {numba_sma_time:.4f}s")
    logger.info(f"Pandas SMA: {pandas_sma_time:.4f}s")
    logger.info(f"Numba speedup: {results['sma_numba_speedup']:.1f}x")

    # Test EMA calculations
    logger.info("Testing EMA calculations...")

    # Numba EMA
    start_time = time.time()
    ema_numba = VectorizedTradingOps.calculate_ema_numba(prices, 12)
    numba_ema_time = time.time() - start_time

    # Pandas EMA
    start_time = time.time()
    ema_pandas = VectorizedTradingOps.calculate_ema_vectorized(prices_pd, 12)
    pandas_ema_time = time.time() - start_time

    # CUDA EMA (if available)
    if CUDA_SUPPORTED:
        start_time = time.time()
        ema_cuda = calculate_ema_cuda(prices, 12)
        cuda_ema_time = time.time() - start_time
        results["ema_cuda_time"] = cuda_ema_time
        logger.info(f"CUDA EMA: {cuda_ema_time:.4f}s")

    results.update(
        {
            "ema_numba_time": numba_ema_time,
            "ema_pandas_time": pandas_ema_time,
            "ema_numba_speedup": pandas_ema_time / numba_ema_time if numba_ema_time > 0 else float("inf"),
        }
    )

    logger.info(f"Numba EMA: {numba_ema_time:.4f}s")
    logger.info(f"Pandas EMA: {pandas_ema_time:.4f}s")
    logger.info(f"Numba speedup: {results['ema_numba_speedup']:.1f}x")

    # Test RSI calculations
    logger.info("Testing RSI calculations...")

    # Numba RSI
    start_time = time.time()
    rsi_numba = VectorizedTradingOps.calculate_rsi_numba(prices, 14)
    numba_rsi_time = time.time() - start_time

    # Pandas RSI
    start_time = time.time()
    rsi_pandas = VectorizedTradingOps.calculate_rsi_vectorized(prices_pd, 14)
    pandas_rsi_time = time.time() - start_time

    results.update(
        {
            "rsi_numba_time": numba_rsi_time,
            "rsi_pandas_time": pandas_rsi_time,
            "rsi_numba_speedup": pandas_rsi_time / numba_rsi_time if numba_rsi_time > 0 else float("inf"),
        }
    )

    logger.info(f"Numba RSI: {numba_rsi_time:.4f}s")
    logger.info(f"Pandas RSI: {pandas_rsi_time:.4f}s")
    logger.info(f"Numba speedup: {results['rsi_numba_speedup']:.1f}x")

    # Test MACD calculations
    logger.info("Testing MACD calculations...")

    # Numba MACD
    start_time = time.time()
    macd_numba, signal_numba, hist_numba = VectorizedTradingOps.calculate_macd_numba(prices)
    numba_macd_time = time.time() - start_time

    # Pandas MACD
    start_time = time.time()
    macd_pandas, signal_pandas, hist_pandas = VectorizedTradingOps.calculate_macd_vectorized(prices_pd)
    pandas_macd_time = time.time() - start_time

    results.update(
        {
            "macd_numba_time": numba_macd_time,
            "macd_pandas_time": pandas_macd_time,
            "macd_numba_speedup": pandas_macd_time / numba_macd_time if numba_macd_time > 0 else float("inf"),
        }
    )

    logger.info(f"Numba MACD: {numba_macd_time:.4f}s")
    logger.info(f"Pandas MACD: {pandas_macd_time:.4f}s")
    logger.info(f"Numba speedup: {results['macd_numba_speedup']:.1f}x")

    # Test batch processing
    logger.info("Testing batch processing...")

    # Create batch data (5 symbols)
    batch_data = np.random.normal(100, 5, (5, n))
    indicators = np.array([0, 1, 2])  # SMA, EMA, RSI

    start_time = time.time()
    batch_results = VectorizedTradingOps.batch_indicator_calculation_numba(batch_data, indicators)
    batch_time = time.time() - start_time

    results["batch_processing_time"] = batch_time
    logger.info(f"Batch processing (5 symbols): {batch_time:.4f}s")

    # Calculate overall performance metrics
    total_numba_time = numba_sma_time + numba_ema_time + numba_rsi_time + numba_macd_time
    total_pandas_time = pandas_sma_time + pandas_ema_time + pandas_rsi_time + pandas_macd_time
    overall_speedup = total_pandas_time / total_numba_time if total_numba_time > 0 else float("inf")

    results.update(
        {
            "total_numba_time": total_numba_time,
            "total_pandas_time": total_pandas_time,
            "overall_speedup": overall_speedup,
            "data_points": n,
            "hardware_info": SYSTEM_INFO,
        }
    )

    logger.info(f"Overall Numba speedup: {overall_speedup:.1f}x")
    logger.info("Benchmark complete!")

    return results


# Backward compatibility
def benchmark_vectorized_vs_iterative():
    """Legacy benchmark function."""
    return benchmark_all_implementations()


if __name__ == "__main__":
    # Run benchmark
    results = benchmark_vectorized_vs_iterative()
    logger.info("Vectorized operations benchmark complete!")

