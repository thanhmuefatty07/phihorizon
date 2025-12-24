#!/usr/bin/env python3
"""
Unit Tests for On-chain Data Loader

Tests for `src/data/onchain_loader.py`
Following test-first development approach per V6.0 Quality Plan.

Requirements:
- pytest
- pytest-mock (for API mocking)
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Import will be available after implementation
# from src.data.onchain_loader import OnChainLoader, OnChainConfig


class TestOnChainConfig:
    """Tests for OnChainConfig dataclass."""
    
    def test_default_config_values(self):
        """Test default configuration values are sensible."""
        from src.data.onchain_loader import OnChainConfig
        
        config = OnChainConfig()
        
        # Verify thresholds
        assert config.whale_accumulation_threshold > 0
        assert config.whale_distribution_threshold < 0
        assert config.exchange_inflow_threshold > 0
        assert config.request_delay >= 0.5  # Rate limiting
        assert config.max_retries >= 1
        assert config.cache_ttl_hours >= 1
    
    def test_config_customization(self):
        """Test that config can be customized."""
        from src.data.onchain_loader import OnChainConfig
        
        config = OnChainConfig(
            whale_accumulation_threshold=0.05,
            request_delay=1.0
        )
        
        assert config.whale_accumulation_threshold == 0.05
        assert config.request_delay == 1.0


class TestOnChainLoader:
    """Tests for OnChainLoader class."""
    
    @pytest.fixture
    def loader(self):
        """Create a loader instance for testing."""
        from src.data.onchain_loader import OnChainLoader
        return OnChainLoader()
    
    def test_loader_initialization(self, loader):
        """Test loader initializes correctly."""
        assert loader is not None
        assert loader.config is not None
        assert hasattr(loader, '_cache')
    
    def test_cache_validity_check(self, loader):
        """Test cache validity checking."""
        # Empty cache should be invalid
        assert not loader._is_cache_valid('nonexistent_key')
        
        # Add to cache
        loader._cache['test_key'] = (datetime.now(), pd.DataFrame())
        assert loader._is_cache_valid('test_key')
        
        # Expired cache should be invalid
        old_time = datetime.now() - timedelta(hours=48)
        loader._cache['old_key'] = (old_time, pd.DataFrame())
        assert not loader._is_cache_valid('old_key')


class TestGlassnodeAPI:
    """Tests for Glassnode API integration."""
    
    @pytest.fixture
    def loader(self):
        from src.data.onchain_loader import OnChainLoader
        return OnChainLoader()
    
    @patch('requests.get')
    def test_get_exchange_netflow_success(self, mock_get, loader):
        """Test successful exchange netflow fetch."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = [
            {'t': 1640000000, 'v': 1000.5},
            {'t': 1640086400, 'v': -500.2},
        ]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Test
        df = loader.get_exchange_netflow(days=2, use_cache=False)
        
        # Verify
        assert df is not None
        assert len(df) == 2
        assert 'timestamp' in df.columns
        assert 'netflow' in df.columns
    
    @patch('requests.get')
    def test_api_failure_handling(self, mock_get, loader):
        """Test graceful handling of API failures."""
        mock_get.side_effect = Exception("API Error")
        
        result = loader.get_exchange_netflow(days=1, use_cache=False)
        
        # Should return None on failure, not raise
        assert result is None
    
    @patch('requests.get')
    def test_rate_limiting(self, mock_get, loader):
        """Test that rate limiting is applied."""
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        import time
        start = time.time()
        
        # Make two requests
        loader._make_request("http://test.com")
        loader._make_request("http://test.com")
        
        elapsed = time.time() - start
        
        # Should have at least one delay
        assert elapsed >= loader.config.request_delay


class TestWhaleAnalysis:
    """Tests for whale activity analysis."""
    
    @pytest.fixture
    def loader(self):
        from src.data.onchain_loader import OnChainLoader
        return OnChainLoader()
    
    def test_classify_whale_activity_accumulation(self, loader):
        """Test whale accumulation classification."""
        # High positive netflow = accumulation
        result = loader.classify_whale_activity(0.10)  # 10% inflow
        assert result == "accumulation"
    
    def test_classify_whale_activity_distribution(self, loader):
        """Test whale distribution classification."""
        # High negative netflow = distribution
        result = loader.classify_whale_activity(-0.10)  # 10% outflow
        assert result == "distribution"
    
    def test_classify_whale_activity_neutral(self, loader):
        """Test neutral whale activity."""
        result = loader.classify_whale_activity(0.01)  # Small movement
        assert result == "neutral"
    
    def test_should_allow_trade_accumulation(self, loader):
        """Test trade allowed during accumulation."""
        allow, reason = loader.should_allow_trade(0.10)
        assert allow is True
        assert "accumulation" in reason.lower()
    
    def test_should_allow_trade_distribution(self, loader):
        """Test trade blocked during distribution."""
        allow, reason = loader.should_allow_trade(-0.10)
        assert allow is False
        assert "distribution" in reason.lower()
    
    def test_should_allow_trade_neutral(self, loader):
        """Test trade allowed during neutral."""
        allow, reason = loader.should_allow_trade(0.01)
        assert allow is True
        assert "neutral" in reason.lower()


class TestDataMerging:
    """Tests for merging on-chain data with OHLCV."""
    
    @pytest.fixture
    def loader(self):
        from src.data.onchain_loader import OnChainLoader
        return OnChainLoader()
    
    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='1H')
        return pd.DataFrame({
            'datetime': dates,
            'open': np.random.uniform(40000, 45000, 10),
            'high': np.random.uniform(45000, 46000, 10),
            'low': np.random.uniform(39000, 40000, 10),
            'close': np.random.uniform(40000, 45000, 10),
            'volume': np.random.uniform(100, 1000, 10),
        })
    
    @pytest.fixture
    def sample_netflow(self):
        """Create sample netflow data."""
        dates = pd.date_range(start='2024-01-01', periods=5, freq='1D')
        return pd.DataFrame({
            'timestamp': dates,
            'netflow': [100, -50, 200, -100, 150],
        })
    
    def test_merge_with_ohlcv(self, loader, sample_ohlcv, sample_netflow):
        """Test merging netflow data with OHLCV."""
        merged = loader.merge_with_ohlcv(sample_ohlcv, sample_netflow)
        
        assert merged is not None
        assert len(merged) == len(sample_ohlcv)
        assert 'whale_netflow' in merged.columns
    
    def test_merge_handles_missing_data(self, loader, sample_ohlcv):
        """Test merge handles None netflow data gracefully."""
        merged = loader.merge_with_ohlcv(sample_ohlcv, None)
        
        assert merged is not None
        assert len(merged) == len(sample_ohlcv)
        # Should have NaN columns
        assert 'whale_netflow' in merged.columns


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    @patch('src.data.onchain_loader.OnChainLoader')
    def test_get_current_whale_activity(self, mock_loader_class):
        """Test get_current_whale_activity convenience function."""
        from src.data.onchain_loader import get_current_whale_activity
        
        mock_loader = Mock()
        mock_loader.get_latest_netflow.return_value = {'netflow': 0.05}
        mock_loader_class.return_value = mock_loader
        
        result = get_current_whale_activity()
        
        assert result is not None
    
    @patch('src.data.onchain_loader.OnChainLoader')
    def test_is_whale_accumulating(self, mock_loader_class):
        """Test is_whale_accumulating convenience function."""
        from src.data.onchain_loader import is_whale_accumulating
        
        mock_loader = Mock()
        mock_loader.get_latest_netflow.return_value = {'netflow': 0.10}
        mock_loader.classify_whale_activity.return_value = "accumulation"
        mock_loader_class.return_value = mock_loader
        
        result = is_whale_accumulating()
        
        assert isinstance(result, bool)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
