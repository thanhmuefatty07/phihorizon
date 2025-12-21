"""
PhiHorizon - Backtesting Module

Advanced backtesting framework with walk-forward optimization.
"""

from .walk_forward import (
    AdvancedWalkForwardOptimizer,
    OptimizationResult,
    WalkForwardConfig,
    optimize_strategy_walk_forward,
)

# Alias for backward compatibility
WalkForwardOptimizer = AdvancedWalkForwardOptimizer

__all__ = [
    "WalkForwardOptimizer",
    "AdvancedWalkForwardOptimizer",
    "WalkForwardConfig",
    "OptimizationResult",
    "optimize_strategy_walk_forward",
]

