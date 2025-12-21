"""
Strategy Tests

Tests for BaseStrategy and RSIStrategy classes.
"""

import pytest
import pandas as pd
import numpy as np


class TestBaseStrategy:
    """Tests for BaseStrategy abstract class."""
    
    def test_import(self):
        """Test that BaseStrategy can be imported."""
        from src.strategy.base import BaseStrategy
        assert BaseStrategy is not None
    
    def test_is_abstract(self):
        """Test that BaseStrategy cannot be instantiated directly."""
        from src.strategy.base import BaseStrategy
        
        # Should raise TypeError because generate_signals is abstract
        with pytest.raises(TypeError):
            BaseStrategy()


class TestRSIStrategy:
    """Tests for RSIStrategy class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data."""
        np.random.seed(42)
        n = 100
        close = 40000 + np.cumsum(np.random.randn(n) * 100)
        
        return pd.DataFrame({
            'open': close - np.random.rand(n) * 50,
            'high': close + np.random.rand(n) * 100,
            'low': close - np.random.rand(n) * 100,
            'close': close,
            'volume': np.random.rand(n) * 1000000
        })
    
    def test_import(self):
        """Test that RSIStrategy can be imported."""
        from src.strategy.base import RSIStrategy
        assert RSIStrategy is not None
    
    def test_init_default(self):
        """Test initialization with defaults."""
        from src.strategy.base import RSIStrategy
        
        strategy = RSIStrategy()
        assert strategy.period == 14
        assert strategy.overbought == 70
        assert strategy.oversold == 30
    
    def test_init_custom(self):
        """Test initialization with custom params."""
        from src.strategy.base import RSIStrategy
        
        strategy = RSIStrategy({'period': 10, 'overbought': 80, 'oversold': 20})
        assert strategy.period == 10
        assert strategy.overbought == 80
        assert strategy.oversold == 20
    
    def test_generate_signals(self, sample_data):
        """Test signal generation."""
        from src.strategy.base import RSIStrategy
        
        strategy = RSIStrategy()
        signals = strategy.generate_signals(sample_data)
        
        assert isinstance(signals, list)
    
    def test_backtest(self, sample_data):
        """Test backtest returns metrics."""
        from src.strategy.base import RSIStrategy
        
        strategy = RSIStrategy()
        metrics = strategy.backtest(sample_data)
        
        assert 'sharpe' in metrics
        assert 'total_return' in metrics
        assert isinstance(metrics['sharpe'], float)


class TestTradeSignal:
    """Tests for TradeSignal dataclass."""
    
    def test_create_signal(self):
        """Test creating a trade signal."""
        from src.strategy.base import TradeSignal
        from datetime import datetime
        
        signal = TradeSignal(
            timestamp=datetime.now(),
            symbol='BTC-USD',
            direction='long',
            strength=0.8
        )
        
        assert signal.direction == 'long'
        assert signal.strength == 0.8
        assert signal.symbol == 'BTC-USD'
