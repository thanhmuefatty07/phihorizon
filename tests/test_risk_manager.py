"""
Risk Manager Tests

Tests for AdvancedRiskManager and risk calculations.
"""

import pytest
import pandas as pd
import numpy as np


class TestRiskLimits:
    """Tests for RiskLimits dataclass."""
    
    def test_default_values(self):
        """Test default risk limits."""
        from src.risk.advanced_risk_manager import RiskLimits
        
        limits = RiskLimits()
        assert limits.max_position_size == 0.1
        assert limits.max_drawdown == 0.2
        assert limits.max_daily_loss == 0.05
        assert limits.stop_loss_pct == 0.02


class TestRiskMetrics:
    """Tests for RiskMetrics dataclass."""
    
    def test_create_metrics(self):
        """Test creating risk metrics."""
        from src.risk.advanced_risk_manager import RiskMetrics
        
        metrics = RiskMetrics(
            var_95=-0.02,
            cvar_95=-0.03,
            sharpe_ratio=1.5,
            max_drawdown=0.1
        )
        
        assert metrics.var_95 == -0.02
        assert metrics.sharpe_ratio == 1.5
    
    def test_to_dict(self):
        """Test converting metrics to dict."""
        from src.risk.advanced_risk_manager import RiskMetrics
        
        metrics = RiskMetrics(var_95=-0.02, sharpe_ratio=1.5)
        d = metrics.to_dict()
        
        assert 'var_95' in d
        assert 'sharpe_ratio' in d


class TestAdvancedRiskManager:
    """Tests for AdvancedRiskManager class."""
    
    @pytest.fixture
    def risk_manager(self):
        """Create a risk manager instance."""
        from src.risk.advanced_risk_manager import AdvancedRiskManager
        return AdvancedRiskManager(portfolio_value=100000)
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns."""
        np.random.seed(42)
        return pd.Series(np.random.randn(252) * 0.02)
    
    def test_import(self):
        """Test that risk manager can be imported."""
        from src.risk import AdvancedRiskManager
        assert AdvancedRiskManager is not None
    
    def test_init(self, risk_manager):
        """Test initialization."""
        assert risk_manager.portfolio_value == 100000
        assert risk_manager.limits is not None
    
    def test_calculate_risk_metrics(self, risk_manager, sample_returns):
        """Test risk metrics calculation."""
        metrics = risk_manager.calculate_risk_metrics(sample_returns)
        
        assert metrics.var_95 < 0  # VaR should be negative
        assert isinstance(metrics.sharpe_ratio, float)
        assert 0 <= metrics.max_drawdown <= 1
    
    def test_calculate_position_size(self, risk_manager):
        """Test position sizing."""
        size = risk_manager.calculate_position_size(
            signal_strength=0.8,
            current_price=40000,
            volatility=0.02
        )
        
        assert size > 0
        assert size <= risk_manager.portfolio_value / 40000  # Max position
    
    def test_check_risk_limits(self, risk_manager, sample_returns):
        """Test risk limit checks."""
        metrics = risk_manager.calculate_risk_metrics(sample_returns)
        checks = risk_manager.check_risk_limits(metrics)
        
        assert 'drawdown_ok' in checks
        assert 'daily_loss_ok' in checks
        assert isinstance(checks['drawdown_ok'], bool)
    
    def test_should_stop_trading(self, risk_manager, sample_returns):
        """Test trading halt decision."""
        metrics = risk_manager.calculate_risk_metrics(sample_returns)
        should_stop = risk_manager.should_stop_trading(metrics)
        
        assert isinstance(should_stop, bool)
