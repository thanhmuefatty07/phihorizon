"""
Unit Tests for Position Sizer and Risk Management

Tests Kelly criterion, volatility sizing, and ATR-based stops.
"""

import numpy as np
import pandas as pd
import pytest


class TestRiskConfig:
    """Tests for RiskConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from src.risk.position_sizer import RiskConfig
        
        config = RiskConfig()
        assert config.kelly_fraction == 0.25
        assert config.max_position_pct == 0.10
        assert config.atr_stop_multiplier == 2.0
        assert config.atr_tp_multiplier == 3.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        from src.risk.position_sizer import RiskConfig
        
        config = RiskConfig(kelly_fraction=0.5, max_position_pct=0.05)
        assert config.kelly_fraction == 0.5
        assert config.max_position_pct == 0.05


class TestPositionSizer:
    """Tests for PositionSizer class."""
    
    def test_import(self):
        """Test module imports correctly."""
        from src.risk.position_sizer import PositionSizer
        assert PositionSizer is not None
    
    def test_create_sizer(self):
        """Test creating sizer instance."""
        from src.risk.position_sizer import PositionSizer
        
        sizer = PositionSizer(account_size=10000)
        assert sizer.account_size == 10000


class TestKellyCriterion:
    """Tests for Kelly criterion calculations."""
    
    def test_kelly_positive_expectancy(self):
        """Test Kelly with positive expectancy."""
        from src.risk.position_sizer import PositionSizer
        
        sizer = PositionSizer(account_size=10000)
        
        # 55% win rate, 2:1 reward ratio
        kelly = sizer.calculate_kelly_fraction(
            win_rate=0.55,
            avg_win=0.02,
            avg_loss=0.01,
        )
        
        # Should be positive
        assert kelly > 0
        # Should be bounded by max
        assert kelly <= sizer.config.max_position_pct
    
    def test_kelly_negative_expectancy(self):
        """Test Kelly with negative expectancy returns 0."""
        from src.risk.position_sizer import PositionSizer
        
        sizer = PositionSizer()
        
        # 30% win rate, 1:1 ratio = negative expectancy
        kelly = sizer.calculate_kelly_fraction(
            win_rate=0.30,
            avg_win=0.01,
            avg_loss=0.01,
        )
        
        # Should be 0 or very small
        assert kelly == 0
    
    def test_kelly_edge_cases(self):
        """Test Kelly with edge case inputs."""
        from src.risk.position_sizer import PositionSizer
        
        sizer = PositionSizer()
        
        # Zero loss
        assert sizer.calculate_kelly_fraction(0.5, 0.01, 0) == 0
        
        # Zero win rate
        assert sizer.calculate_kelly_fraction(0, 0.01, 0.01) == 0
        
        # 100% win rate
        assert sizer.calculate_kelly_fraction(1, 0.01, 0.01) == 0


class TestPositionCalculation:
    """Tests for position size calculation."""
    
    def test_calculate_position_size(self):
        """Test calculating position size."""
        from src.risk.position_sizer import PositionSizer
        
        sizer = PositionSizer(account_size=10000)
        
        position = sizer.calculate_position_size(
            win_rate=0.55,
            avg_win=0.02,
            avg_loss=0.01,
            current_price=100,
        )
        
        assert 'kelly_fraction' in position
        assert 'position_dollars' in position
        assert 'position_units' in position
        assert position['position_dollars'] <= 10000 * 0.10  # Max 10%
    
    def test_volatility_position_high_vol(self):
        """High volatility should result in smaller or equal position."""
        from src.risk.position_sizer import PositionSizer
        
        sizer = PositionSizer()
        
        low_vol_position = sizer.calculate_volatility_position(volatility=0.01)
        high_vol_position = sizer.calculate_volatility_position(volatility=0.05)
        
        # High vol should give smaller or equal position
        # (may be equal if both hit max cap)
        assert high_vol_position <= low_vol_position


class TestStopsAndTargets:
    """Tests for stop loss and take profit calculations."""
    
    def test_stop_loss_calculation(self):
        """Test stop loss calculation."""
        from src.risk.position_sizer import PositionSizer
        
        sizer = PositionSizer()
        
        stop = sizer.calculate_stop_loss(
            atr=100,
            entry_price=10000,
            direction='long',
        )
        
        assert 'stop_distance' in stop
        assert 'stop_price' in stop
        assert stop['stop_price'] == 10000 - (100 * 2)  # 2x ATR below entry
    
    def test_take_profit_calculation(self):
        """Test take profit calculation."""
        from src.risk.position_sizer import PositionSizer
        
        sizer = PositionSizer()
        
        tp = sizer.calculate_take_profit(
            atr=100,
            entry_price=10000,
            direction='long',
        )
        
        assert 'tp_distance' in tp
        assert 'tp_price' in tp
        assert 'risk_reward' in tp
        assert tp['tp_price'] == 10000 + (100 * 3)  # 3x ATR above entry
        assert tp['risk_reward'] == 1.5  # 3/2
    
    def test_short_direction(self):
        """Test stops/TP for short direction."""
        from src.risk.position_sizer import PositionSizer
        
        sizer = PositionSizer()
        
        stop = sizer.calculate_stop_loss(atr=100, entry_price=10000, direction='short')
        tp = sizer.calculate_take_profit(atr=100, entry_price=10000, direction='short')
        
        # Short: stop above, TP below
        assert stop['stop_price'] == 10000 + 200
        assert tp['tp_price'] == 10000 - 300


class TestATRCalculation:
    """Tests for ATR calculation."""
    
    def test_calculate_atr(self):
        """Test ATR calculation."""
        from src.risk.position_sizer import PositionSizer
        
        sizer = PositionSizer()
        
        np.random.seed(42)
        high = pd.Series(100 + np.random.rand(50) * 2)
        low = pd.Series(98 + np.random.rand(50) * 2)
        close = pd.Series(99 + np.random.rand(50) * 2)
        
        atr = sizer.calculate_atr(high, low, close, period=14)
        
        assert len(atr) == 50
        assert atr.iloc[-1] > 0


class TestTradePlan:
    """Tests for complete trade plan."""
    
    def test_get_trade_plan(self):
        """Test generating complete trade plan."""
        from src.risk.position_sizer import PositionSizer
        
        sizer = PositionSizer(account_size=10000)
        
        plan = sizer.get_trade_plan(
            entry_price=10000,
            atr=100,
            win_rate=0.55,
            avg_win=0.02,
            avg_loss=0.01,
            direction='long',
        )
        
        assert 'entry_price' in plan
        assert 'position' in plan
        assert 'stop_loss' in plan
        assert 'take_profit' in plan
        assert 'risk_reward' in plan
        assert 'max_loss_dollars' in plan


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_required_win_rate(self):
        """Test calculating required win rate."""
        from src.risk.position_sizer import calculate_required_win_rate
        
        # 1:1 ratio requires 50%
        assert calculate_required_win_rate(0.01, 0.01) == 0.5
        
        # 2:1 ratio requires 33%
        rate = calculate_required_win_rate(0.02, 0.01)
        assert abs(rate - 0.333) < 0.01
    
    def test_expectancy(self):
        """Test expectancy calculation."""
        from src.risk.position_sizer import calculate_expectancy
        
        # Positive expectancy
        exp = calculate_expectancy(win_rate=0.55, avg_win=0.02, avg_loss=0.01)
        assert exp > 0
        
        # Negative expectancy
        exp = calculate_expectancy(win_rate=0.30, avg_win=0.01, avg_loss=0.01)
        assert exp < 0
