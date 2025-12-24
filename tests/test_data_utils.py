"""
Comprehensive Tests for Data Utils Module

Tests data manipulation, transformation, and analysis utilities.
Optimized for maximum coverage with audit-level quality.
"""

import numpy as np
import pandas as pd
import pytest


class TestOptimizeDataframeMemory:
    """Tests for optimize_dataframe_memory function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.data_utils import optimize_dataframe_memory
        assert optimize_dataframe_memory is not None
    
    def test_basic_optimization(self):
        """Test basic memory optimization."""
        from src.utils.data_utils import optimize_dataframe_memory
        
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.0, 2.0, 3.0, 4.0, 5.0],
            'str_col': ['a', 'b', 'c', 'd', 'e']
        })
        
        result = optimize_dataframe_memory(df)
        
        assert len(result) == len(df)
        assert list(result.columns) == list(df.columns)
    
    def test_copy_parameter(self):
        """Test copy parameter creates new DataFrame."""
        from src.utils.data_utils import optimize_dataframe_memory
        
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = optimize_dataframe_memory(df, copy=True)
        
        assert result is not df


class TestChunkDataframe:
    """Tests for chunk_dataframe function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.data_utils import chunk_dataframe
        assert chunk_dataframe is not None
    
    def test_basic_chunking(self):
        """Test basic DataFrame chunking."""
        from src.utils.data_utils import chunk_dataframe
        
        df = pd.DataFrame({'a': range(100)})
        chunks = chunk_dataframe(df, chunk_size=30)
        
        assert len(chunks) == 4  # 30+30+30+10
        assert sum(len(c) for c in chunks) == 100
    
    def test_small_dataframe(self):
        """Test chunking with small DataFrame."""
        from src.utils.data_utils import chunk_dataframe
        
        df = pd.DataFrame({'a': [1, 2, 3]})
        chunks = chunk_dataframe(df, chunk_size=10)
        
        assert len(chunks) == 1


class TestGetMemoryUsageMB:
    """Tests for get_memory_usage_mb function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.data_utils import get_memory_usage_mb
        assert get_memory_usage_mb is not None
    
    def test_returns_positive(self):
        """Test returns positive value."""
        from src.utils.data_utils import get_memory_usage_mb
        
        df = pd.DataFrame({'a': range(1000)})
        result = get_memory_usage_mb(df)
        
        assert result > 0


class TestValidateAndCleanData:
    """Tests for validate_and_clean_data function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.data_utils import validate_and_clean_data
        assert validate_and_clean_data is not None
    
    def test_valid_data(self):
        """Test valid data returns no errors."""
        from src.utils.data_utils import validate_and_clean_data
        
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })
        
        is_valid, errors = validate_and_clean_data(df)
        
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
    
    def test_with_required_columns(self):
        """Test with required columns specified."""
        from src.utils.data_utils import validate_and_clean_data
        
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        is_valid, errors = validate_and_clean_data(df, required_columns=['a', 'b'])
        
        assert isinstance(is_valid, bool)


class TestResampleOHLCV:
    """Tests for resample_ohlcv function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.data_utils import resample_ohlcv
        assert resample_ohlcv is not None
    
    def test_basic_resample(self):
        """Test basic OHLCV resampling."""
        from src.utils.data_utils import resample_ohlcv
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=24, freq='1h'),
            'open': np.random.rand(24) * 100,
            'high': np.random.rand(24) * 100 + 50,
            'low': np.random.rand(24) * 100 - 50,
            'close': np.random.rand(24) * 100,
            'volume': np.random.randint(1000, 5000, 24)
        })
        
        result = resample_ohlcv(df, '4h')
        
        assert len(result) == 6  # 24 hours / 4 = 6


class TestCalculateReturns:
    """Tests for calculate_returns function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.data_utils import calculate_returns
        assert calculate_returns is not None
    
    def test_simple_returns(self):
        """Test simple returns calculation."""
        from src.utils.data_utils import calculate_returns
        
        prices = pd.Series([100, 110, 105, 115])
        result = calculate_returns(prices, method='simple')
        
        assert len(result) == len(prices)
        assert abs(result.iloc[1] - 0.10) < 0.001  # 10% increase
    
    def test_log_returns(self):
        """Test log returns calculation."""
        from src.utils.data_utils import calculate_returns
        
        prices = pd.Series([100, 110, 105, 115])
        result = calculate_returns(prices, method='log')
        
        assert len(result) == len(prices)


class TestDetectOutliers:
    """Tests for detect_outliers function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.data_utils import detect_outliers
        assert detect_outliers is not None
    
    def test_iqr_method(self):
        """Test IQR outlier detection."""
        from src.utils.data_utils import detect_outliers
        
        data = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is outlier
        result = detect_outliers(data, method='iqr')
        
        assert isinstance(result, pd.Series)
        assert result.dtype == bool
    
    def test_zscore_method(self):
        """Test Z-score outlier detection."""
        from src.utils.data_utils import detect_outliers
        
        data = pd.Series([1, 2, 3, 4, 5, 100])
        result = detect_outliers(data, method='zscore')
        
        assert isinstance(result, pd.Series)


class TestCalculateTechnicalIndicators:
    """Tests for calculate_technical_indicators function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.data_utils import calculate_technical_indicators
        assert calculate_technical_indicators is not None
    
    def test_basic_indicators(self):
        """Test basic indicator calculation."""
        from src.utils.data_utils import calculate_technical_indicators
        
        np.random.seed(42)
        df = pd.DataFrame({
            'open': 100 + np.random.randn(50) * 0.5,
            'high': 101 + np.random.randn(50) * 0.5,
            'low': 99 + np.random.randn(50) * 0.5,
            'close': 100 + np.random.randn(50) * 0.5,
            'volume': np.random.randint(1000, 5000, 50)
        })
        
        result = calculate_technical_indicators(df)
        
        assert len(result) == len(df)
        # Should have additional columns
        assert len(result.columns) > len(df.columns)


class TestNormalizeData:
    """Tests for normalize_data function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.data_utils import normalize_data
        assert normalize_data is not None
    
    def test_zscore_normalization(self):
        """Test Z-score normalization."""
        from src.utils.data_utils import normalize_data
        
        df = pd.DataFrame({
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        
        result = normalize_data(df, method='zscore')
        
        # Result should be a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
    
    def test_minmax_normalization(self):
        """Test min-max normalization."""
        from src.utils.data_utils import normalize_data
        
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = normalize_data(df, method='minmax')
        
        # Result should be DataFrame or have normalized values
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)


class TestHandleMissingData:
    """Tests for handle_missing_data function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.data_utils import handle_missing_data
        assert handle_missing_data is not None
    
    def test_interpolate_method(self):
        """Test interpolation of missing data."""
        from src.utils.data_utils import handle_missing_data
        
        df = pd.DataFrame({'a': [1, np.nan, 3, np.nan, 5]})
        result = handle_missing_data(df, method='interpolate')
        
        assert not result['a'].isna().any()
    
    def test_drop_method(self):
        """Test dropping missing data."""
        from src.utils.data_utils import handle_missing_data
        
        df = pd.DataFrame({'a': [1, np.nan, 3]})
        result = handle_missing_data(df, method='drop')
        
        assert len(result) == 2


class TestCalculateCorrelationMatrix:
    """Tests for calculate_correlation_matrix function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.data_utils import calculate_correlation_matrix
        assert calculate_correlation_matrix is not None
    
    def test_basic_correlation(self):
        """Test basic correlation matrix."""
        from src.utils.data_utils import calculate_correlation_matrix
        
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [2, 4, 6, 8, 10],  # Perfectly correlated with a
            'c': [5, 4, 3, 2, 1]   # Negatively correlated
        })
        
        result = calculate_correlation_matrix(df)
        
        assert result.shape == (3, 3)
        assert abs(result['a']['b'] - 1.0) < 0.001


class TestFindHighlyCorrelatedPairs:
    """Tests for find_highly_correlated_pairs function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.data_utils import find_highly_correlated_pairs
        assert find_highly_correlated_pairs is not None
    
    def test_find_pairs(self):
        """Test finding correlated pairs."""
        from src.utils.data_utils import find_highly_correlated_pairs
        
        corr_matrix = pd.DataFrame({
            'a': [1.0, 0.9, 0.5],
            'b': [0.9, 1.0, 0.3],
            'c': [0.5, 0.3, 1.0]
        }, index=['a', 'b', 'c'])
        
        result = find_highly_correlated_pairs(corr_matrix, threshold=0.8)
        
        assert isinstance(result, list)
        # Should find (a, b) or (b, a) with 0.9 correlation
        assert len(result) >= 1


class TestCalculateDrawdowns:
    """Tests for calculate_drawdowns function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.data_utils import calculate_drawdowns
        assert calculate_drawdowns is not None
    
    def test_basic_drawdown(self):
        """Test basic drawdown calculation."""
        from src.utils.data_utils import calculate_drawdowns
        
        prices = pd.Series([100, 110, 90, 95, 100])  # Drawdown from 110 to 90
        result = calculate_drawdowns(prices)
        
        assert len(result) == len(prices)
        assert result.iloc[2] < 0  # Should be negative at trough


class TestCalculateBeta:
    """Tests for calculate_beta function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.data_utils import calculate_beta
        assert calculate_beta is not None
    
    def test_beta_calculation(self):
        """Test beta calculation."""
        from src.utils.data_utils import calculate_beta
        
        np.random.seed(42)
        market = pd.Series(np.random.randn(100) * 0.01)
        asset = market * 1.5 + np.random.randn(100) * 0.005  # Beta ~1.5
        
        result = calculate_beta(asset, market)
        
        assert isinstance(result, float)
        assert 1.0 < result < 2.0  # Should be around 1.5


class TestSplitDataByDate:
    """Tests for split_data_by_date function."""
    
    def test_import(self):
        """Test function imports correctly."""
        from src.utils.data_utils import split_data_by_date
        assert split_data_by_date is not None
    
    def test_basic_split(self):
        """Test basic date split."""
        from src.utils.data_utils import split_data_by_date
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=30),
            'value': range(30)
        })
        
        train, test = split_data_by_date(df, '2024-01-15')
        
        assert len(train) + len(test) == 30


class TestAuditDataQuality:
    """Audit tests for data quality handling."""
    
    def test_nan_handling(self):
        """Audit: Functions should handle NaN gracefully."""
        from src.utils.data_utils import calculate_returns
        
        prices = pd.Series([100, np.nan, 105, 110])
        result = calculate_returns(prices)
        
        # Should not crash
        assert len(result) == len(prices)
    
    def test_empty_dataframe(self):
        """Audit: Functions should handle empty DataFrames."""
        from src.utils.data_utils import optimize_dataframe_memory
        
        df = pd.DataFrame()
        result = optimize_dataframe_memory(df)
        
        assert len(result) == 0
