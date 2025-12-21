"""
Core functionality tests for PhiHorizon.

These tests verify that the main components work correctly.
"""

import pytest
import numpy as np
import pandas as pd


class TestWalkForwardOptimizer:
    """Tests for Walk-Forward Optimization."""
    
    def test_import(self):
        """Test that WalkForwardOptimizer can be imported."""
        from src.backtesting import WalkForwardOptimizer
        assert WalkForwardOptimizer is not None
    
    def test_config_creation(self):
        """Test WalkForwardConfig creation."""
        from src.backtesting import WalkForwardConfig
        config = WalkForwardConfig()
        assert config.in_sample_periods == 252
        assert config.out_sample_periods == 63


class TestPhiCalculator:
    """Tests for Consciousness Phi metrics."""
    
    def test_import(self):
        """Test that PhiCalculator can be imported."""
        from src.consciousness.metrics import PhiCalculator
        assert PhiCalculator is not None
    
    def test_phi_calculation(self, sample_ohlcv_data):
        """Test Phi calculation on sample data."""
        from src.consciousness.metrics import PhiCalculator
        
        calculator = PhiCalculator()
        phi = calculator.calculate(sample_ohlcv_data)
        
        # Phi should be a valid float
        assert isinstance(phi, float)
        assert 0 <= phi <= 1


class TestBaseStrategy:
    """Tests for Base Strategy."""
    
    def test_import(self):
        """Test that BaseStrategy can be imported."""
        from src.strategy.base import BaseStrategy, RSIStrategy
        assert BaseStrategy is not None
        assert RSIStrategy is not None
    
    def test_rsi_strategy(self, sample_ohlcv_data):
        """Test RSI Strategy generates signals."""
        from src.strategy.base import RSIStrategy
        
        strategy = RSIStrategy({'period': 14, 'overbought': 70, 'oversold': 30})
        signals = strategy.generate_signals(sample_ohlcv_data)
        
        assert isinstance(signals, list)

