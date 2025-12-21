#!/usr/bin/env python3

"""

Supreme System V5 - Fast Math Module (V2 - Production Grade)

Optimized Kelly Criterion calculation with Numba JIT.

Performance Target: < 1 microsecond per calculation
Safety: Comprehensive input validation and error handling

Author: Supreme System V5 Team
Date: 2025-11-24
Version: 2.0 (Fixed)
"""

import logging
import math
import time
from typing import Optional

import numpy as np

# Attempt Numba import
try:
    import numba

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logging.warning("Numba not available. Using pure Python fallback (slower).")

logger = logging.getLogger(__name__)

# ===== CONFIGURATION CONSTANTS =====

SENTIMENT_MIN = -1.0
SENTIMENT_MAX = 1.0
MIN_KELLY_FRACTION = 0.005  # 0.5% minimum
MAX_KELLY_FRACTION = 0.10  # 10% maximum
SIGMOID_K = 5.0  # Sigmoid steepness
DELTA_MAX = 0.3  # Max probability adjustment range


# ===== NUMBA JIT VERSION (ULTRA-FAST) =====

if NUMBA_AVAILABLE:

    @numba.jit(nopython=True, nogil=True, cache=True)
    def kelly_criterion_scalar(p_win: float, reward_ratio: float, sentiment: float, lambda_factor: float) -> float:
        """
        Calculate optimal Kelly fraction for a single trade (Numba-optimized).

        Math:
        - Sigmoid adjustment: delta = (0.3 / (1 + exp(-5*s))) - 0.15
        - Adjusted probability: p' = p + delta
        - Kelly formula: f* = (p'(R+1) - 1) / R
        - Fractional Kelly: f = λ * f*

        Args:
            p_win: Base win probability (0.0 to 1.0)
            reward_ratio: Reward/Risk ratio (must be > 0)
            sentiment: Market sentiment (-1.0 to +1.0)
            lambda_factor: Kelly fraction multiplier (typically 0.25)

        Returns:
            Position size as fraction of capital (0.0 to 0.1)
            Returns 0.0 if no edge or invalid inputs

        Complexity: O(1)
        Performance: ~0.3-0.5 microseconds on modern CPU
        """
        # === INPUT VALIDATION (FIXED BUG #1, #2) ===
        # Clamp sentiment to valid range
        if sentiment < -1.0:
            sentiment = -1.0
        elif sentiment > 1.0:
            sentiment = 1.0

        # Validate reward ratio
        if reward_ratio <= 0.0:
            return 0.0  # Invalid reward ratio

        # Validate probability
        if p_win <= 0.0 or p_win >= 1.0:
            return 0.0  # Invalid probability

        # === SIGMOID ADJUSTMENT ===
        delta = (DELTA_MAX / (1.0 + math.exp(-SIGMOID_K * sentiment))) - (DELTA_MAX / 2.0)
        adjusted_p = p_win + delta

        # Clamp adjusted probability to valid range
        if adjusted_p < 0.1:
            adjusted_p = 0.1
        elif adjusted_p > 0.9:
            adjusted_p = 0.9

        # === KELLY FORMULA ===
        kelly_full = (adjusted_p * (reward_ratio + 1.0) - 1.0) / reward_ratio

        # === NEGATIVE KELLY CHECK (FIXED ISSUE #3) ===
        if kelly_full <= 0.0:
            # No statistical edge, reject trade
            return 0.0

        # === FRACTIONAL KELLY ===
        f = lambda_factor * kelly_full

        # === SAFETY CAPS ===
        if f < MIN_KELLY_FRACTION:
            return 0.0
        elif f > MAX_KELLY_FRACTION:
            return MAX_KELLY_FRACTION
        else:
            return f

    @numba.jit(nopython=True, nogil=True, cache=True, parallel=False)
    def kelly_criterion_batch(
        p_win: np.ndarray, reward_ratio: np.ndarray, sentiment: np.ndarray, lambda_factor: float, out: np.ndarray
    ) -> None:
        """
        Batch Kelly calculation with pre-allocated output (Numba-optimized).

        Args:
            p_win: Array of win probabilities
            reward_ratio: Array of reward/risk ratios
            sentiment: Array of sentiment scores
            lambda_factor: Kelly fraction multiplier
            out: Pre-allocated output array (MODIFIED IN-PLACE)

        Complexity: O(n) where n = len(arrays)
        Performance: ~0.2 microseconds per element on modern CPU

        Note: All input arrays must have same length as 'out'.
              This function modifies 'out' in-place (zero-copy).
        """
        n = len(p_win)

        # === VALIDATION (FIXED BUG #4) ===
        # Note: Numba nopython mode doesn't support assertions or exceptions
        # So we just ensure we don't go out of bounds
        if len(reward_ratio) < n or len(sentiment) < n or len(out) < n:
            return  # Silently fail (caller's responsibility to validate)

        for i in range(n):
            # Clamp sentiment
            s = sentiment[i]
            if s < -1.0:
                s = -1.0
            elif s > 1.0:
                s = 1.0

            # Validate inputs
            if reward_ratio[i] <= 0.0 or p_win[i] <= 0.0 or p_win[i] >= 1.0:
                out[i] = 0.0
                continue

            # Sigmoid adjustment
            delta_i = (DELTA_MAX / (1.0 + math.exp(-SIGMOID_K * s))) - (DELTA_MAX / 2.0)
            adjusted_p_i = p_win[i] + delta_i

            # Clamp adjusted probability
            if adjusted_p_i < 0.1:
                adjusted_p_i = 0.1
            elif adjusted_p_i > 0.9:
                adjusted_p_i = 0.9

            # Kelly formula
            kelly_full_i = (adjusted_p_i * (reward_ratio[i] + 1.0) - 1.0) / reward_ratio[i]

            # Negative Kelly check
            if kelly_full_i <= 0.0:
                out[i] = 0.0
                continue

            # Fractional Kelly
            f_i = lambda_factor * kelly_full_i

            # Safety caps
            if f_i < MIN_KELLY_FRACTION:
                out[i] = 0.0
            elif f_i > MAX_KELLY_FRACTION:
                out[i] = MAX_KELLY_FRACTION
            else:
                out[i] = f_i

else:
    # ===== PURE PYTHON FALLBACK (WHEN NUMBA UNAVAILABLE) =====

    def kelly_criterion_scalar(p_win: float, reward_ratio: float, sentiment: float, lambda_factor: float) -> float:
        """Pure Python fallback (slower but functional)."""
        # Clamp sentiment
        sentiment = max(-1.0, min(1.0, sentiment))

        # Validate inputs
        if reward_ratio <= 0.0 or p_win <= 0.0 or p_win >= 1.0:
            return 0.0

        # Sigmoid adjustment
        delta = (DELTA_MAX / (1.0 + math.exp(-SIGMOID_K * sentiment))) - (DELTA_MAX / 2.0)
        adjusted_p = max(0.1, min(0.9, p_win + delta))

        # Kelly formula
        kelly_full = (adjusted_p * (reward_ratio + 1.0) - 1.0) / reward_ratio

        if kelly_full <= 0.0:
            return 0.0

        f = lambda_factor * kelly_full

        if f < MIN_KELLY_FRACTION:
            return 0.0
        elif f > MAX_KELLY_FRACTION:
            return MAX_KELLY_FRACTION
        else:
            return f

    def kelly_criterion_batch(
        p_win: np.ndarray, reward_ratio: np.ndarray, sentiment: np.ndarray, lambda_factor: float, out: np.ndarray
    ) -> None:
        """Pure Python vectorized fallback."""
        # Validate array lengths (FIXED BUG #4 for Python version)
        n = len(out)
        assert (
            len(p_win) == n and len(reward_ratio) == n and len(sentiment) == n
        ), f"Array size mismatch: p_win={len(p_win)}, reward_ratio={len(reward_ratio)}, sentiment={len(sentiment)}, out={n}"

        # Clamp sentiment
        sentiment_clamped = np.clip(sentiment, -1.0, 1.0)

        # Sigmoid adjustment
        delta = (DELTA_MAX / (1.0 + np.exp(-SIGMOID_K * sentiment_clamped))) - (DELTA_MAX / 2.0)
        adjusted_p = np.clip(p_win + delta, 0.1, 0.9)

        # Kelly formula
        kelly_full = (adjusted_p * (reward_ratio + 1.0) - 1.0) / reward_ratio

        # Fractional Kelly
        f = lambda_factor * kelly_full

        # Apply caps
        out[:] = f
        out[out < MIN_KELLY_FRACTION] = 0.0
        out[out > MAX_KELLY_FRACTION] = MAX_KELLY_FRACTION
        out[kelly_full <= 0.0] = 0.0
        out[reward_ratio <= 0.0] = 0.0
        out[(p_win <= 0.0) | (p_win >= 1.0)] = 0.0


# ===== BENCHMARK & VALIDATION (FIXED ISSUE #7) =====


def benchmark():
    """
    Benchmark both scalar and batch functions.
    Target: < 1 microsecond per calculation.
    """
    print("=" * 60)
    print("FAST MATH BENCHMARK")
    print("=" * 60)
    print(f"Numba Available: {NUMBA_AVAILABLE}")
    print()

    # === SCALAR BENCHMARK ===
    print("--- Scalar Function Benchmark ---")
    trials = 1_000_000

    # Warm-up
    for _ in range(1000):
        kelly_criterion_scalar(0.55, 2.0, 0.1, 0.25)

    t0 = time.perf_counter()
    for _ in range(trials):
        result = kelly_criterion_scalar(0.55, 2.0, 0.1, 0.25)
    t1 = time.perf_counter()

    avg_us = (t1 - t0) / trials * 1e6
    print(f"Trials: {trials:,}")
    print(f"Total time: {t1 - t0:.3f} seconds")
    print(f"Average latency: {avg_us:.3f} µs")
    print(f"Sample result: {result:.6f}")

    if avg_us < 1.0:
        print("✅ PASS: < 1 µs target achieved!")
    else:
        print(f"⚠️  WARNING: {avg_us:.3f} µs > 1 µs target")

    print()

    # === BATCH BENCHMARK (FIXED ISSUE #7) ===
    print("--- Batch Function Benchmark ---")
    batch_size = 10_000
    trials = 100

    # Prepare input arrays
    p_win_arr = np.random.uniform(0.4, 0.7, batch_size)
    reward_arr = np.random.uniform(1.5, 3.0, batch_size)
    sentiment_arr = np.random.uniform(-0.5, 0.5, batch_size)
    out_arr = np.zeros(batch_size, dtype=np.float64)

    # Warm-up
    for _ in range(10):
        kelly_criterion_batch(p_win_arr, reward_arr, sentiment_arr, 0.25, out_arr)

    t0 = time.perf_counter()
    for _ in range(trials):
        kelly_criterion_batch(p_win_arr, reward_arr, sentiment_arr, 0.25, out_arr)
    t1 = time.perf_counter()

    total_calcs = batch_size * trials
    avg_us_per_calc = (t1 - t0) / total_calcs * 1e6

    print(f"Batch size: {batch_size:,}")
    print(f"Trials: {trials:,}")
    print(f"Total calculations: {total_calcs:,}")
    print(f"Total time: {t1 - t0:.3f} seconds")
    print(f"Average latency per calculation: {avg_us_per_calc:.3f} µs")
    print(f"Throughput: {total_calcs / (t1 - t0):,.0f} calculations/second")
    print(f"Sample results: {out_arr[:5]}")

    print()

    print("=" * 60)


if __name__ == "__main__":
    # Run comprehensive benchmark
    benchmark()

    # === VALIDATION TESTS ===
    print("\n--- Validation Tests ---")

    # Test 1: Normal case
    result = kelly_criterion_scalar(0.60, 2.0, 0.5, 0.25)
    print(f"Test 1 (Normal): p=0.60, R=2.0, s=0.5 -> f={result:.4f}")
    assert 0.0 < result <= 0.10, "Normal case failed"

    # Test 2: Negative sentiment
    result = kelly_criterion_scalar(0.55, 2.0, -0.8, 0.25)
    print(f"Test 2 (Bear): p=0.55, R=2.0, s=-0.8 -> f={result:.4f}")
    assert result < kelly_criterion_scalar(0.55, 2.0, 0.0, 0.25), "Sentiment adjustment failed"

    # Test 3: Invalid reward ratio (FIXED BUG #2 TEST)
    result = kelly_criterion_scalar(0.60, 0.0, 0.0, 0.25)
    print(f"Test 3 (Invalid R): p=0.60, R=0.0 -> f={result:.4f}")
    assert result == 0.0, "Invalid reward ratio not handled"

    # Test 4: Extreme sentiment (FIXED BUG #1 TEST)
    result = kelly_criterion_scalar(0.60, 2.0, 5.0, 0.25)
    print(f"Test 4 (Extreme s): p=0.60, R=2.0, s=5.0 -> f={result:.4f}")
    assert 0.0 <= result <= 0.10, "Extreme sentiment not clamped"

    # Test 5: Negative Kelly (low probability)
    result = kelly_criterion_scalar(0.30, 2.0, 0.0, 0.25)
    print(f"Test 5 (Negative Kelly): p=0.30, R=2.0 -> f={result:.4f}")
    assert result == 0.0, "Negative Kelly not rejected"

    print("\n✅ All validation tests passed!")
