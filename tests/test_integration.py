"""
Integration Tests

End-to-end tests verifying component integration.
"""

import pytest
import pandas as pd
import numpy as np


class TestCoreImports:
    """Test all core imports work together."""
    
    def test_all_core_imports(self):
        """Test that all core modules can be imported."""
        from src.backtesting import WalkForwardOptimizer, AdvancedWalkForwardOptimizer
        from src.backtesting.production_backtester import ProductionBacktester
        from src.consciousness.metrics import PhiCalculator, IITCore
        from src.risk import AdvancedRiskManager
        from src.strategy.base import BaseStrategy, RSIStrategy
        
        assert all([
            WalkForwardOptimizer,
            AdvancedWalkForwardOptimizer,
            ProductionBacktester,
            PhiCalculator,
            IITCore,
            AdvancedRiskManager,
            BaseStrategy,
            RSIStrategy
        ])


class TestWorkflowIntegration:
    """Test typical workflow integration."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data for testing."""
        np.random.seed(42)
        n = 500
        dates = pd.date_range('2024-01-01', periods=n, freq='1h')
        close = 40000 + np.cumsum(np.random.randn(n) * 100)
        
        return pd.DataFrame({
            'open': close - np.random.rand(n) * 50,
            'high': close + np.random.rand(n) * 100,
            'low': close - np.random.rand(n) * 100,
            'close': close,
            'volume': np.random.rand(n) * 1000000
        }, index=dates)
    
    def test_strategy_to_phi(self, sample_data):
        """Test strategy generates signals and phi calculates."""
        from src.strategy.base import RSIStrategy
        from src.consciousness.metrics import PhiCalculator
        
        # Generate signals
        strategy = RSIStrategy()
        signals = strategy.generate_signals(sample_data)
        
        # Calculate phi
        phi_calc = PhiCalculator()
        phi = phi_calc.calculate_phi(sample_data)
        
        assert isinstance(signals, list)
        assert 0 <= phi <= 1
    
    def test_risk_manager_with_returns(self, sample_data):
        """Test risk manager calculates metrics from data."""
        from src.risk import AdvancedRiskManager
        
        returns = sample_data['close'].pct_change().dropna()
        
        risk_mgr = AdvancedRiskManager(portfolio_value=100000)
        metrics = risk_mgr.calculate_risk_metrics(returns)
        
        assert metrics.var_95 != 0
        assert isinstance(metrics.sharpe_ratio, float)


class TestBenchmarkReportFormat:
    """Test benchmark report format compatibility."""
    
    def test_report_structure(self):
        """Test benchmark_report.json has expected structure."""
        import json
        import os
        
        report_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'benchmark_report.json'
        )
        
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            # Check required sections
            assert 'meta' in report
            assert 'walk_forward' in report
            assert 'xgboost' in report
            assert 'consciousness' in report
            assert 'conclusion' in report
            
            # Check conclusion has pass/fail flags
            assert 'accuracy_pass' in report['conclusion']
            assert 'ready_for_production' in report['conclusion']
