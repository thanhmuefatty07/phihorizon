"""
Backtesting Risk Manager for Supreme System V5.

Simplified risk management for backtesting only.
No real-time cooldown, no trade frequency limits.

Hardware: Designed for i5 8th gen + 4GB RAM (laptop)
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BacktestRiskConfig:
    """Risk configuration for backtesting."""

    max_position_pct: float = 0.10  # Max 10% of portfolio per trade
    max_drawdown_pct: float = 0.05  # Max 5% portfolio drawdown
    max_total_exposure_pct: float = 0.50  # Max 50% total exposure


@dataclass
class RiskCheckResult:
    """Result of risk check."""

    approved: bool
    adjusted_size: Optional[float] = None
    reason: str = ""


class BacktestRiskManager:
    """
    Simplified risk manager for backtesting.

    Only checks:
    - Position size limits
    - Maximum drawdown
    - Total exposure

    No real-time features like:
    - Cooldown periods
    - Trade frequency limits
    - Circuit breakers
    """

    def __init__(self, config: Optional[BacktestRiskConfig] = None):
        """
        Initialize backtesting risk manager.

        Args:
            config: Risk configuration
        """
        self.config = config or BacktestRiskConfig()

        # Backtesting state
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.current_exposure = 0.0

        logger.info(
            f"BacktestRiskManager initialized: "
            f"max_position={self.config.max_position_pct*100}%, "
            f"max_drawdown={self.config.max_drawdown_pct*100}%"
        )

    def check_trade(self, position_size_pct: float, current_equity: float) -> RiskCheckResult:
        """
        Check if trade is allowed.

        Args:
            position_size_pct: Position size as percentage of portfolio
            current_equity: Current portfolio equity

        Returns:
            RiskCheckResult with approval status
        """
        # Update equity tracking
        self.current_equity = current_equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Check max drawdown
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            if drawdown >= self.config.max_drawdown_pct:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Max drawdown exceeded: {drawdown*100:.1f}% >= {self.config.max_drawdown_pct*100}%",
                )

        # Check position size
        if position_size_pct > self.config.max_position_pct:
            adjusted = self.config.max_position_pct
            return RiskCheckResult(
                approved=True,
                adjusted_size=adjusted,
                reason=f"Position size reduced: {position_size_pct*100:.1f}% -> {adjusted*100}%",
            )

        # Check total exposure
        new_exposure = self.current_exposure + position_size_pct
        if new_exposure > self.config.max_total_exposure_pct:
            return RiskCheckResult(
                approved=False,
                reason=f"Total exposure would exceed limit: {new_exposure*100:.1f}% > {self.config.max_total_exposure_pct*100}%",
            )

        return RiskCheckResult(approved=True, reason="Trade approved")

    def update_exposure(self, position_change_pct: float):
        """Update current exposure after trade."""
        self.current_exposure += position_change_pct
        self.current_exposure = max(0, self.current_exposure)

    def get_current_drawdown(self) -> float:
        """Calculate current drawdown percentage."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity

    def reset(self):
        """Reset for new backtesting run."""
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.current_exposure = 0.0
        logger.info("BacktestRiskManager reset")
