"""
PhiHorizon - AI Trading Research Framework

A production-ready framework for backtesting and ML-based trading research.
"""

__version__ = "5.0.0"
__author__ = "Supreme Trading Team"
__description__ = "AI Trading Research Framework with Walk-Forward and Consciousness Metrics"

# Core modules only - import from packages, not modules directly
from .backtesting import WalkForwardOptimizer, AdvancedWalkForwardOptimizer
from .consciousness.metrics import PhiCalculator, IITCore

__all__ = [
    "WalkForwardOptimizer",
    "AdvancedWalkForwardOptimizer",
    "PhiCalculator",
    "IITCore",
]

