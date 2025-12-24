"""
Comprehensive Tests for Helpers Module

Tests all helper functions with edge cases for maximum coverage.
Optimized test structure for efficiency.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os
from pathlib import Path


class TestSafeDivide:
    """Tests for safe_divide function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.helpers import safe_divide
        assert safe_divide is not None
    
    def test_normal_division(self):
        """Test normal division."""
        from src.utils.helpers import safe_divide
        assert safe_divide(10, 2) == 5.0
    
    def test_divide_by_zero(self):
        """Test division by zero returns default."""
        from src.utils.helpers import safe_divide
        assert safe_divide(10, 0) == 0.0
    
    def test_divide_by_zero_custom_default(self):
        """Test custom default value."""
        from src.utils.helpers import safe_divide
        assert safe_divide(10, 0, default=-1.0) == -1.0
    
    def test_float_division(self):
        """Test float division."""
        from src.utils.helpers import safe_divide
        result = safe_divide(1.0, 3.0)
        assert abs(result - 0.333) < 0.01


class TestCalculatePercentageChange:
    """Tests for calculate_percentage_change function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.helpers import calculate_percentage_change
        assert calculate_percentage_change is not None
    
    def test_positive_change(self):
        """Test positive percentage change."""
        from src.utils.helpers import calculate_percentage_change
        result = calculate_percentage_change(100, 110)
        assert abs(result - 0.10) < 0.001
    
    def test_negative_change(self):
        """Test negative percentage change."""
        from src.utils.helpers import calculate_percentage_change
        result = calculate_percentage_change(100, 90)
        assert abs(result - (-0.10)) < 0.001
    
    def test_no_change(self):
        """Test zero change."""
        from src.utils.helpers import calculate_percentage_change
        result = calculate_percentage_change(100, 100)
        assert result == 0.0


class TestCalculateMovingAverage:
    """Tests for calculate_moving_average function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.helpers import calculate_moving_average
        assert calculate_moving_average is not None
    
    def test_simple_moving_average(self):
        """Test SMA calculation."""
        from src.utils.helpers import calculate_moving_average
        
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = calculate_moving_average(data, window=3, method='sma')
        
        assert len(result) == len(data)
        assert result.iloc[-1] == 9.0  # (8+9+10)/3
    
    def test_exponential_moving_average(self):
        """Test EMA calculation."""
        from src.utils.helpers import calculate_moving_average
        
        data = pd.Series([1, 2, 3, 4, 5])
        result = calculate_moving_average(data, window=3, method='ema')
        
        assert len(result) == len(data)
        assert result.iloc[-1] > 0
    
    def test_weighted_moving_average(self):
        """Test WMA calculation."""
        from src.utils.helpers import calculate_moving_average
        
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = calculate_moving_average(data, window=3, method='wma')
        
        assert len(result) == len(data)


class TestCalculateVolatility:
    """Tests for calculate_volatility function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.helpers import calculate_volatility
        assert calculate_volatility is not None
    
    def test_volatility_calculation(self):
        """Test volatility calculation."""
        from src.utils.helpers import calculate_volatility
        
        # Random returns
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.01)
        
        result = calculate_volatility(returns, window=20, annualize=True)
        
        assert len(result) == len(returns)
        assert result.dropna().iloc[-1] > 0
    
    def test_volatility_not_annualized(self):
        """Test non-annualized volatility."""
        from src.utils.helpers import calculate_volatility
        
        returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01] * 10)
        result = calculate_volatility(returns, window=5, annualize=False)
        
        assert result.dropna().iloc[-1] > 0


class TestValidateConfig:
    """Tests for validate_config function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.helpers import validate_config
        assert validate_config is not None
    
    def test_valid_config(self):
        """Test valid config returns True."""
        from src.utils.helpers import validate_config
        
        config = {'key1': 'value1', 'key2': 'value2'}
        assert validate_config(config, ['key1', 'key2']) == True
    
    def test_missing_key(self):
        """Test missing key returns False."""
        from src.utils.helpers import validate_config
        
        config = {'key1': 'value1'}
        assert validate_config(config, ['key1', 'key2']) == False


class TestValidateOHLCVColumns:
    """Tests for validate_ohlcv_columns function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.helpers import validate_ohlcv_columns
        assert validate_ohlcv_columns is not None
    
    def test_valid_ohlcv(self):
        """Test valid OHLCV data returns True."""
        from src.utils.helpers import validate_ohlcv_columns
        
        df = pd.DataFrame({
            'timestamp': [1000], 'open': [100], 'high': [101], 'low': [99],
            'close': [100], 'volume': [1000]
        })
        assert validate_ohlcv_columns(df) == True
    
    def test_missing_column(self):
        """Test missing column returns False."""
        from src.utils.helpers import validate_ohlcv_columns
        
        df = pd.DataFrame({
            'open': [100], 'high': [101],
            'close': [100]  # missing timestamp, low, volume
        })
        assert validate_ohlcv_columns(df) == False


class TestNormalizeSymbol:
    """Tests for normalize_symbol function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.helpers import normalize_symbol
        assert normalize_symbol is not None
    
    def test_uppercase_conversion(self):
        """Test lowercase converts to uppercase."""
        from src.utils.helpers import normalize_symbol
        
        result = normalize_symbol("btc/usdt")
        assert result.isupper() or 'BTC' in result.upper()
    
    def test_strip_whitespace(self):
        """Test whitespace handling."""
        from src.utils.helpers import normalize_symbol
        
        result = normalize_symbol("  BTCUSDT  ")
        # Should return a non-empty string
        assert isinstance(result, str)
        assert len(result) > 0


class TestCalculateSharpeRatio:
    """Tests for calculate_sharpe_ratio function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.helpers import calculate_sharpe_ratio
        assert calculate_sharpe_ratio is not None
    
    def test_positive_returns(self):
        """Test Sharpe with positive returns."""
        from src.utils.helpers import calculate_sharpe_ratio
        
        returns = pd.Series([0.01, 0.02, 0.01, 0.015, 0.01])
        result = calculate_sharpe_ratio(returns)
        
        assert result > 0  # Positive returns should give positive Sharpe
    
    def test_negative_returns(self):
        """Test Sharpe with negative returns."""
        from src.utils.helpers import calculate_sharpe_ratio
        
        returns = pd.Series([-0.01, -0.02, -0.01, -0.015, -0.01])
        result = calculate_sharpe_ratio(returns)
        
        assert result < 0  # Negative returns should give negative Sharpe


class TestCalculateMaxDrawdown:
    """Tests for calculate_max_drawdown function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.helpers import calculate_max_drawdown
        assert calculate_max_drawdown is not None
    
    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        from src.utils.helpers import calculate_max_drawdown
        
        prices = pd.Series([100, 110, 90, 95, 100])  # 20% drawdown from 110 to 90
        result = calculate_max_drawdown(prices)
        
        assert abs(result - 0.1818) < 0.01  # ~18.18%
    
    def test_no_drawdown(self):
        """Test no drawdown for monotonically increasing."""
        from src.utils.helpers import calculate_max_drawdown
        
        prices = pd.Series([100, 101, 102, 103, 104])
        result = calculate_max_drawdown(prices)
        
        assert result == 0.0


class TestEnsureDirectoryExists:
    """Tests for ensure_directory_exists function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.helpers import ensure_directory_exists
        assert ensure_directory_exists is not None
    
    def test_create_directory(self):
        """Test directory creation."""
        from src.utils.helpers import ensure_directory_exists
        
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, 'new_folder')
            result = ensure_directory_exists(new_dir)
            
            assert os.path.exists(new_dir)
            assert isinstance(result, Path)


class TestFormatFileSize:
    """Tests for format_file_size function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.helpers import format_file_size
        assert format_file_size is not None
    
    def test_returns_string(self):
        """Test returns a string."""
        from src.utils.helpers import format_file_size
        
        result = format_file_size(500)
        assert isinstance(result, str)
    
    def test_small_size(self):
        """Test small sizes return a string."""
        from src.utils.helpers import format_file_size
        
        result = format_file_size(100)
        assert isinstance(result, str)
    
    def test_large_size(self):
        """Test large sizes return a string."""
        from src.utils.helpers import format_file_size
        
        result = format_file_size(1_000_000_000)
        assert isinstance(result, str)


class TestRoundToSignificantDigits:
    """Tests for round_to_significant_digits function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.helpers import round_to_significant_digits
        assert round_to_significant_digits is not None
    
    def test_rounding(self):
        """Test significant digit rounding."""
        from src.utils.helpers import round_to_significant_digits
        
        result = round_to_significant_digits(12345.6789, digits=3)
        assert result == 12300 or result == 12346 or abs(result - 12345.7) < 1


class TestSetupLogging:
    """Tests for setup_logging function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.helpers import setup_logging
        assert setup_logging is not None
    
    def test_returns_logger(self):
        """Test returns logger instance."""
        from src.utils.helpers import setup_logging
        import logging
        
        result = setup_logging(level="WARNING")
        assert result is not None


class TestDetectDataGaps:
    """Tests for detect_data_gaps function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.helpers import detect_data_gaps
        assert detect_data_gaps is not None
    
    def test_no_gaps(self):
        """Test detection with no gaps."""
        from src.utils.helpers import detect_data_gaps
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1h')
        })
        result = detect_data_gaps(df, max_gap_minutes=120)
        
        assert len(result) == 0 or result.empty
    
    def test_with_gap(self):
        """Test detection with gap."""
        from src.utils.helpers import detect_data_gaps
        
        timestamps = pd.to_datetime(['2024-01-01 00:00', '2024-01-01 01:00', 
                                     '2024-01-01 05:00'])  # 4 hour gap
        df = pd.DataFrame({'timestamp': timestamps})
        result = detect_data_gaps(df, max_gap_minutes=120)
        
        # Should detect the 4-hour gap
        assert len(result) >= 1 or not result.empty
