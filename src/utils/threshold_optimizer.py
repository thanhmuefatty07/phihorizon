#!/usr/bin/env python3
"""
PhiHorizon - Threshold Optimization Utilities

Research-backed threshold optimization for trading classification:
1. F0.5-optimal threshold (precision-focused)
2. Sharpe-optimal threshold 
3. Cost-sensitive threshold selection

Source: Precision-Recall tradeoff research in financial ML
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def find_optimal_threshold_f05(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_trades: int = 10,
    threshold_range: Tuple[float, float] = (0.3, 0.95),
    n_thresholds: int = 50,
) -> Tuple[float, float, Dict]:
    """
    Find threshold that maximizes F0.5 score (precision-focused).
    
    F0.5 weights precision 2x more than recall.
    Use when: False positives are more costly than false negatives.
    In trading: A bad trade (FP) is worse than missing a good trade (FN).
    
    Research: Cost-sensitive classification in finance papers.
    
    Args:
        y_true: True labels (0/1)
        y_proba: Predicted probabilities
        min_trades: Minimum predictions required (avoid trivial thresholds)
        threshold_range: (min, max) threshold to search
        n_thresholds: Number of thresholds to evaluate
        
    Returns:
        (optimal_threshold, best_f05_score, metrics_dict)
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    if len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must have same length")
    
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
    
    best_f05 = 0.0
    best_thresh = 0.5
    all_metrics = []
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        n_positive = y_pred.sum()
        
        if n_positive < min_trades:
            continue
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # F0.5 = 1.25 * (P * R) / (0.25 * P + R)
        if precision + recall > 0:
            f05 = 1.25 * (precision * recall) / (0.25 * precision + recall)
        else:
            f05 = 0.0
        
        metrics = {
            'threshold': t,
            'f05': f05,
            'precision': precision,
            'recall': recall,
            'n_trades': int(n_positive),
            'accuracy': accuracy_score(y_true, y_pred),
        }
        all_metrics.append(metrics)
        
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = t
    
    # Find metrics at best threshold
    best_metrics = next(
        (m for m in all_metrics if m['threshold'] == best_thresh),
        {'threshold': best_thresh, 'f05': best_f05}
    )
    
    logger.info(f"F0.5-optimal threshold: {best_thresh:.2f} (F0.5={best_f05:.4f})")
    
    return best_thresh, best_f05, best_metrics


def find_optimal_threshold_f1(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_trades: int = 10,
) -> Tuple[float, float]:
    """
    Find threshold that maximizes F1 score (balanced).
    
    Use when: Equal importance for precision and recall.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        min_trades: Minimum predictions required
        
    Returns:
        (optimal_threshold, best_f1_score)
    """
    thresholds = np.linspace(0.3, 0.9, 50)
    
    best_f1 = 0.0
    best_thresh = 0.5
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        
        if y_pred.sum() < min_trades:
            continue
        
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    
    return best_thresh, best_f1


def find_optimal_threshold_profit(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    returns: np.ndarray,
    fee_rate: float = 0.001,
    min_trades: int = 10,
) -> Tuple[float, float, Dict]:
    """
    Find threshold that maximizes expected profit.
    
    Directly optimizes for P&L instead of accuracy.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        returns: Actual returns per sample
        fee_rate: Transaction cost per trade
        min_trades: Minimum predictions required
        
    Returns:
        (optimal_threshold, best_profit, metrics_dict)
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    returns = np.asarray(returns)
    
    thresholds = np.linspace(0.3, 0.95, 50)
    
    best_profit = -np.inf
    best_thresh = 0.5
    best_metrics = {}
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        n_trades = y_pred.sum()
        
        if n_trades < min_trades:
            continue
        
        # Calculate profit: only trade when predicted positive
        trade_returns = returns * y_pred
        fees = np.abs(trade_returns) * fee_rate * 2  # Entry + exit fees
        net_returns = trade_returns - fees
        
        total_profit = net_returns.sum()
        sharpe = net_returns.mean() / (net_returns.std() + 1e-9) * np.sqrt(252 * 24)
        
        if total_profit > best_profit:
            best_profit = total_profit
            best_thresh = t
            best_metrics = {
                'threshold': t,
                'total_profit': total_profit,
                'sharpe': sharpe,
                'n_trades': int(n_trades),
                'avg_return': net_returns[y_pred == 1].mean() if n_trades > 0 else 0,
                'win_rate': (net_returns[y_pred == 1] > 0).mean() if n_trades > 0 else 0,
            }
    
    return best_thresh, best_profit, best_metrics


def analyze_threshold_sensitivity(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    returns: Optional[np.ndarray] = None,
) -> List[Dict]:
    """
    Analyze model performance across all thresholds.
    
    Returns detailed metrics for each threshold for visualization.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        returns: Optional returns for profit calculation
        
    Returns:
        List of metrics dictionaries for each threshold
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    thresholds = np.linspace(0.1, 0.95, 85)
    results = []
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        n_trades = y_pred.sum()
        
        if n_trades < 5:
            continue
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # Multiple F-scores
        f05 = 1.25 * (precision * recall) / (0.25 * precision + recall) if (precision + recall) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f2 = 5 * (precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'threshold': float(t),
            'n_trades': int(n_trades),
            'precision': float(precision),
            'recall': float(recall),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'f05': float(f05),
            'f1': float(f1),
            'f2': float(f2),
        }
        
        if returns is not None:
            returns = np.asarray(returns)
            trade_returns = returns * y_pred
            metrics['total_return'] = float(trade_returns.sum())
            metrics['sharpe'] = float(
                trade_returns.mean() / (trade_returns.std() + 1e-9) * np.sqrt(252 * 24)
            )
        
        results.append(metrics)
    
    return results


def suggest_threshold_for_regime(
    volatility: float,
    avg_return: float,
    base_threshold: float = 0.5,
) -> float:
    """
    Dynamically adjust threshold based on market regime.
    
    High volatility → Higher threshold (be more selective)
    Low returns → Higher threshold (need more confidence)
    
    Args:
        volatility: Current market volatility
        avg_return: Average recent returns
        base_threshold: Starting threshold
        
    Returns:
        Adjusted threshold
    """
    # Normalize volatility (assuming 0.02 = 2% daily vol is typical)
    vol_factor = volatility / 0.02
    
    # Increase threshold in high volatility
    vol_adjustment = 0.1 * (vol_factor - 1.0)  # +10% per 1x above normal
    
    # Increase threshold in low return environment
    return_adjustment = 0.0
    if avg_return < 0:
        return_adjustment = 0.1  # Be more cautious in negative markets
    
    adjusted = base_threshold + vol_adjustment + return_adjustment
    
    # Bound to reasonable range
    return float(np.clip(adjusted, 0.4, 0.9))


__all__ = [
    'find_optimal_threshold_f05',
    'find_optimal_threshold_f1',
    'find_optimal_threshold_profit',
    'analyze_threshold_sensitivity',
    'suggest_threshold_for_regime',
]
