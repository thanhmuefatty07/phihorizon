"""
PhiHorizon - Consciousness Metrics Module

This module implements IIT (Integrated Information Theory) inspired metrics
for trading signal filtering.

The key insight: Phi measures "integration" in the market data stream.
- High Phi = Market is coherent/tradeable → signals more reliable
- Low Phi = Market is fragmented/noisy → avoid trading

This is a SIMPLIFIED proxy for IIT Phi, not the full mathematical formulation
(which would require PyPhi and be computationally prohibitive for trading).

Based on: Tononi, G. et al. "Integrated Information Theory" (2004-2024)
"""

from .metrics import (
    PhiCalculator,
    IITCore,
    calculate_phi_proxy,
    MarketIntegrationMetrics,
)

from .entropy_metrics import (
    calculate_transfer_entropy,
    calculate_mutual_information,
    calculate_complexity,
)

__all__ = [
    "PhiCalculator",
    "IITCore",
    "calculate_phi_proxy",
    "MarketIntegrationMetrics",
    "calculate_transfer_entropy",
    "calculate_mutual_information",
    "calculate_complexity",
]
