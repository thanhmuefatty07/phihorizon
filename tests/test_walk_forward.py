"""
Walk-Forward Optimizer Tests

Tests for the core walk-forward optimization functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestWalkForwardConfig:
    """Tests for WalkForwardConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        from src.backtesting import WalkForwardConfig
        
        config = WalkForwardConfig()
        assert config.in_sample_periods == 252
        assert config.out_sample_periods == 63
        assert config.step_size == 21
        assert config.min_samples == 3
        assert config.min_sharpe_ratio == 1.0
        assert config.max_drawdown_limit == 0.20
    
    def test_custom_values(self):
        """Test custom configuration values."""
        from src.backtesting import WalkForwardConfig
        
        config = WalkForwardConfig(
            in_sample_periods=126,
            out_sample_periods=21,
            min_samples=12
        )
        assert config.in_sample_periods == 126
        assert config.out_sample_periods == 21
        assert config.min_samples == 12


class TestWalkForwardOptimizer:
    """Tests for AdvancedWalkForwardOptimizer class."""
    
    def test_import(self):
        """Test that optimizer can be imported."""
        from src.backtesting import WalkForwardOptimizer, AdvancedWalkForwardOptimizer
        assert WalkForwardOptimizer is not None
        assert AdvancedWalkForwardOptimizer is not None
    
    def test_init_default_config(self):
        """Test optimizer initialization with default config."""
        from src.backtesting import AdvancedWalkForwardOptimizer
        
        optimizer = AdvancedWalkForwardOptimizer()
        assert optimizer.config is not None
        assert optimizer.config.in_sample_periods == 252
    
    def test_init_custom_config(self):
        """Test optimizer initialization with custom config."""
        from src.backtesting import AdvancedWalkForwardOptimizer, WalkForwardConfig
        
        config = WalkForwardConfig(in_sample_periods=100)
        optimizer = AdvancedWalkForwardOptimizer(config=config)
        assert optimizer.config.in_sample_periods == 100


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""
    
    def test_result_creation(self):
        """Test creating an optimization result."""
        from src.backtesting.walk_forward import OptimizationResult
        
        result = OptimizationResult(
            window_idx=0,
            in_sample_start=datetime.now(),
            in_sample_end=datetime.now(),
            out_sample_start=datetime.now(),
            out_sample_end=datetime.now(),
            optimal_params={'period': 14},
            in_sample_metrics={'sharpe': 1.5},
            out_sample_metrics={'sharpe': 1.0},
            validation_score=0.8,
            overfitting_risk=0.2,
            statistical_significance=0.05
        )
        
        assert result.window_idx == 0
        assert result.optimal_params['period'] == 14
        assert result.validation_score == 0.8
