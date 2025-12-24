#!/usr/bin/env python3
"""
Unit Tests for Funding Rate Loader

Tests for `src/data/funding_loader.py`
Following test-first development approach per V6.0 Quality Plan.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np


class TestFundingConfig:
    """Tests for FundingConfig dataclass."""
    
    def test_default_config_values(self):
        """Test default configuration values are sensible."""
        from src.data.funding_loader import FundingConfig
        
        config = FundingConfig()
        
        # Verify thresholds
        assert config.positive_threshold > 0  # Overleveraged long
        assert config.negative_threshold < 0  # Overleveraged short
        assert config.request_delay >= 0.1
        assert config.max_retries >= 1


class TestFundingLoader:
    """Tests for FundingLoader class."""
    
    @pytest.fixture
    def loader(self):
        """Create a loader instance for testing."""
        from src.data.funding_loader import FundingLoader
        return FundingLoader()
    
    def test_loader_initialization(self, loader):
        """Test loader initializes correctly."""
        assert loader is not None
        assert loader.config is not None
    
    @patch('requests.get')
    def test_get_funding_rate_success(self, mock_get, loader):
        """Test successful funding rate fetch."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'data': [
                {'fundingTime': '1700000000000', 'fundingRate': '0.0001'},
                {'fundingTime': '1700028800000', 'fundingRate': '-0.0002'},
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        df = loader.get_funding_rate(days=1, use_cache=False)
        
        assert df is not None
        assert 'funding_rate' in df.columns
    
    def test_classify_funding_positive(self, loader):
        """Test positive funding rate classification."""
        # High positive = overleveraged long = bearish
        result = loader.classify_funding_rate(0.0015)  # 0.15%
        assert result == "overleveraged_long"
    
    def test_classify_funding_negative(self, loader):
        """Test negative funding rate classification."""
        # High negative = overleveraged short = bullish
        result = loader.classify_funding_rate(-0.0015)
        assert result == "overleveraged_short"
    
    def test_classify_funding_neutral(self, loader):
        """Test neutral funding rate."""
        result = loader.classify_funding_rate(0.0001)
        assert result == "neutral"
    
    def test_should_allow_trade_positive(self, loader):
        """Test trade blocked during overleveraged long."""
        allow, reason = loader.should_allow_trade(0.0015)
        assert allow is False
    
    def test_should_allow_trade_negative(self, loader):
        """Test trade allowed during overleveraged short (contrarian)."""
        allow, reason = loader.should_allow_trade(-0.0015)
        assert allow is True
    
    def test_should_allow_trade_neutral(self, loader):
        """Test trade allowed during neutral."""
        allow, reason = loader.should_allow_trade(0.0001)
        assert allow is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
