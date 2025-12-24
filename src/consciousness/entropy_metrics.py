#!/usr/bin/env python3
"""
PhiHorizon - Entropy-Based Metrics for Phi Calculation

Provides additional entropy-based measures for the consciousness module:
- Transfer Entropy: Directional information flow
- Mutual Information: Shared information between series
- Complexity Measures: Market structure indicators

These are used as building blocks for the Phi calculation in metrics.py.
"""

import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    n_bins: int = 4,
) -> float:
    """
    Calculate Transfer Entropy from source to target.
    
    Transfer Entropy measures the directed information flow from
    source time series to target time series.
    
    TE(X→Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})
    
    If TE(X→Y) > 0, X "causes" Y in information-theoretic sense.
    
    Args:
        source: Source time series
        target: Target time series  
        lag: Time lag for the transfer
        n_bins: Number of bins for discretization
        
    Returns:
        Transfer entropy value (≥0)
    """
    source = np.asarray(source).flatten()
    target = np.asarray(target).flatten()
    
    if len(source) != len(target):
        raise ValueError("Source and target must have same length")
    
    n = len(source)
    if n < lag + 10:
        return 0.0
    
    # Discretize
    def discretize(data):
        if len(np.unique(data)) < n_bins:
            return data.astype(int) % n_bins
        bins = np.percentile(data, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            return np.zeros(len(data), dtype=int)
        return np.clip(np.digitize(data, bins[1:-1]), 0, n_bins - 1)
    
    source_d = discretize(source)
    target_d = discretize(target)
    
    # Build joint distributions
    # P(Y_t, Y_{t-1})
    # P(Y_t, Y_{t-1}, X_{t-1})
    
    # Simple estimation using co-occurrence counting
    eps = 1e-10
    
    # H(Y_t | Y_{t-1})
    joint_yy = np.zeros((n_bins, n_bins))
    for i in range(lag, n):
        y_prev = target_d[i - lag]
        y_curr = target_d[i]
        if 0 <= y_prev < n_bins and 0 <= y_curr < n_bins:
            joint_yy[y_prev, y_curr] += 1
    
    joint_yy /= (joint_yy.sum() + eps)
    p_y_prev = joint_yy.sum(axis=1)
    
    h_y_given_yprev = 0
    for i in range(n_bins):
        if p_y_prev[i] > eps:
            p_y_curr_given_yprev = joint_yy[i] / (p_y_prev[i] + eps)
            mask = p_y_curr_given_yprev > eps
            if mask.any():
                h_y_given_yprev -= p_y_prev[i] * np.sum(
                    p_y_curr_given_yprev[mask] * np.log2(p_y_curr_given_yprev[mask])
                )
    
    # H(Y_t | Y_{t-1}, X_{t-1})
    joint_yyx = np.zeros((n_bins, n_bins, n_bins))
    for i in range(lag, n):
        y_prev = target_d[i - lag]
        x_prev = source_d[i - lag]
        y_curr = target_d[i]
        if 0 <= y_prev < n_bins and 0 <= x_prev < n_bins and 0 <= y_curr < n_bins:
            joint_yyx[y_prev, x_prev, y_curr] += 1
    
    joint_yyx /= (joint_yyx.sum() + eps)
    p_yx_prev = joint_yyx.sum(axis=2)
    
    h_y_given_yxprev = 0
    for i in range(n_bins):
        for j in range(n_bins):
            if p_yx_prev[i, j] > eps:
                p_y_curr_given_yxprev = joint_yyx[i, j] / (p_yx_prev[i, j] + eps)
                mask = p_y_curr_given_yxprev > eps
                if mask.any():
                    h_y_given_yxprev -= p_yx_prev[i, j] * np.sum(
                        p_y_curr_given_yxprev[mask] * np.log2(p_y_curr_given_yxprev[mask])
                    )
    
    # TE = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})
    te = max(0, h_y_given_yprev - h_y_given_yxprev)
    
    return te


def calculate_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 4,
) -> float:
    """
    Calculate Mutual Information between two time series.
    
    I(X;Y) = H(X) + H(Y) - H(X,Y)
    
    Measures the shared information between X and Y.
    
    Args:
        x: First time series
        y: Second time series
        n_bins: Number of bins for discretization
        
    Returns:
        Mutual information in bits
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    if len(x) != len(y):
        raise ValueError("X and Y must have same length")
    
    if len(x) < 10:
        return 0.0
    
    # Discretize
    def discretize(data):
        if len(np.unique(data)) < n_bins:
            return data.astype(int) % n_bins
        bins = np.percentile(data, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            return np.zeros(len(data), dtype=int)
        return np.clip(np.digitize(data, bins[1:-1]), 0, n_bins - 1)
    
    x_d = discretize(x)
    y_d = discretize(y)
    
    # Calculate distributions
    eps = 1e-10
    
    p_x = np.bincount(x_d, minlength=n_bins) / len(x_d)
    p_y = np.bincount(y_d, minlength=n_bins) / len(y_d)
    
    # Joint distribution
    joint = np.zeros((n_bins, n_bins))
    for i in range(len(x_d)):
        if 0 <= x_d[i] < n_bins and 0 <= y_d[i] < n_bins:
            joint[x_d[i], y_d[i]] += 1
    joint /= (joint.sum() + eps)
    
    # Entropies
    def entropy(p):
        p = p[p > eps]
        return -np.sum(p * np.log2(p))
    
    h_x = entropy(p_x)
    h_y = entropy(p_y)
    h_xy = entropy(joint.flatten())
    
    # I(X;Y) = H(X) + H(Y) - H(X,Y)
    mi = max(0, h_x + h_y - h_xy)
    
    return mi


def calculate_complexity(
    data: np.ndarray,
    method: str = 'lempel_ziv',
) -> float:
    """
    Calculate complexity measure of a time series.
    
    Complexity captures the "interesting" structure in data:
    - Low complexity: Random or highly deterministic
    - High complexity: Rich, structured patterns
    
    Args:
        data: Time series data
        method: Complexity measure ('lempel_ziv', 'sample_entropy', 'approximate')
        
    Returns:
        Normalized complexity (0-1)
    """
    data = np.asarray(data).flatten()
    
    if len(data) < 10:
        return 0.5
    
    if method == 'lempel_ziv':
        return _lempel_ziv_complexity(data)
    elif method == 'sample_entropy':
        return _sample_entropy(data)
    else:  # approximate
        return _approximate_complexity(data)


def _lempel_ziv_complexity(data: np.ndarray) -> float:
    """
    Calculate Lempel-Ziv complexity (LZ76).
    
    Measures the number of distinct patterns in the data.
    """
    # Binarize: above/below median
    binary = (data > np.median(data)).astype(int)
    
    # Count distinct substrings
    n = len(binary)
    if n == 0:
        return 0.5
    
    s = ''.join(map(str, binary))
    
    # Simple LZ complexity: count new patterns
    complexity = 1
    i = 0
    k = 1
    
    while i + k <= n:
        substring = s[i:i+k]
        # Check if this substring appeared before
        if substring in s[:i]:
            k += 1
        else:
            complexity += 1
            i = i + k
            k = 1
    
    # Normalize by theoretical upper bound
    # Upper bound ≈ n / log2(n)
    if n > 1:
        upper_bound = n / np.log2(n)
        normalized = min(1.0, complexity / upper_bound)
    else:
        normalized = 0.5
    
    return normalized


def _sample_entropy(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Calculate Sample Entropy.
    
    Measures the unpredictability of the time series.
    Lower entropy = more regular/predictable
    """
    n = len(data)
    if n < m + 2:
        return 0.5
    
    r_scaled = r * np.std(data)
    if r_scaled < 1e-10:
        return 0.5
    
    # Count matching template pairs
    def count_matches(data, m, r_val):
        n = len(data) - m
        count = 0
        for i in range(n):
            template = data[i:i+m]
            for j in range(i+1, n):
                if np.max(np.abs(template - data[j:j+m])) < r_val:
                    count += 1
        return count
    
    try:
        a = count_matches(data, m, r_scaled)
        b = count_matches(data, m+1, r_scaled)
        
        if a == 0 or b == 0:
            return 0.5
        
        sampen = -np.log(b / a)
        
        # Normalize (typical range 0-3)
        normalized = 1 - np.exp(-sampen)
        return np.clip(normalized, 0, 1)
        
    except Exception:
        return 0.5


def _approximate_complexity(data: np.ndarray) -> float:
    """
    Fast approximate complexity using autocorrelation structure.
    """
    if len(data) < 20:
        return 0.5
    
    # Calculate autocorrelation at multiple lags
    max_lag = min(20, len(data) // 3)
    autocorrs = []
    
    data_centered = data - np.mean(data)
    var = np.var(data_centered)
    
    if var < 1e-10:
        return 0.0
    
    for lag in range(1, max_lag):
        if lag < len(data):
            corr = np.correlate(data_centered[:-lag], data_centered[lag:])[0]
            corr /= (len(data) - lag) * var
            autocorrs.append(abs(corr))
    
    if len(autocorrs) < 2:
        return 0.5
    
    autocorrs = np.array(autocorrs)
    
    # Complexity = diversity of autocorrelation structure
    # Random: all autocorrs ≈ 0 → low complexity
    # Deterministic: smooth decay → low complexity  
    # Complex: varied pattern → high complexity
    
    mean_ac = np.mean(autocorrs)
    std_ac = np.std(autocorrs)
    
    # Combine mean and variability
    complexity = (mean_ac + std_ac) / 2
    
    return np.clip(complexity * 3, 0, 1)  # Scale up


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'calculate_transfer_entropy',
    'calculate_mutual_information',
    'calculate_complexity',
]
