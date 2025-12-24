"""
Unit Tests for CCXT Loader (OKX)

Tests for OKX data loading, pagination, and hybrid Kaggle+OKX pipeline.
"""

import numpy as np
import pandas as pd
import pytest


class TestOKXConfig:
    """Tests for OKXConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from src.data.ccxt_loader import OKXConfig
        
        config = OKXConfig()
        assert config.symbol == "BTC/USDT"
        assert config.timeframe == "1h"
        assert config.max_candles_per_request == 1000
    
    def test_custom_config(self):
        """Test custom configuration."""
        from src.data.ccxt_loader import OKXConfig
        
        config = OKXConfig(
            symbol="ETH/USDT",
            timeframe="4h",
            max_candles_per_request=500,
        )
        assert config.symbol == "ETH/USDT"
        assert config.timeframe == "4h"


class TestCCXTLoader:
    """Tests for CCXTLoader class."""
    
    def test_import(self):
        """Test module imports correctly."""
        from src.data.ccxt_loader import CCXTLoader
        assert CCXTLoader is not None
    
    def test_create_loader(self):
        """Test creating loader instance."""
        from src.data.ccxt_loader import CCXTLoader, OKXConfig
        
        config = OKXConfig()
        loader = CCXTLoader(config)
        assert loader is not None
    
    def test_is_available_property(self):
        """Test is_available property."""
        from src.data.ccxt_loader import CCXTLoader
        
        loader = CCXTLoader()
        # May be True or False depending on CCXT installation
        assert isinstance(loader.is_available, bool)
    
    def test_get_exchange_info(self):
        """Test exchange info retrieval."""
        from src.data.ccxt_loader import CCXTLoader
        
        loader = CCXTLoader()
        info = loader.get_exchange_info()
        
        # Should return dict (with info or error)
        assert isinstance(info, dict)


class TestKaggleOKXHybridLoader:
    """Tests for KaggleOKXHybridLoader class."""
    
    def test_import(self):
        """Test module imports correctly."""
        from src.data.ccxt_loader import KaggleOKXHybridLoader
        assert KaggleOKXHybridLoader is not None
    
    def test_create_hybrid_loader(self):
        """Test creating hybrid loader instance."""
        from src.data.ccxt_loader import KaggleOKXHybridLoader
        
        loader = KaggleOKXHybridLoader()
        assert loader is not None
        assert loader.okx_loader is not None
    
    def test_load_csv_standardization(self):
        """Test CSV standardization logic."""
        from src.data.ccxt_loader import KaggleOKXHybridLoader
        import tempfile
        import os
        
        loader = KaggleOKXHybridLoader()
        
        # Create temp CSV with non-standard columns
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("DateTime,Open,High,Low,Price,Vol\n")
            f.write("2024-01-01,100,102,99,101,1000\n")
            f.write("2024-01-02,101,103,100,102,1100\n")
            temp_path = f.name
        
        try:
            df = loader._load_csv(temp_path)
            
            # Should have standardized columns
            assert 'timestamp' in df.columns or 'datetime' in df.columns
            assert 'close' in df.columns
            assert len(df) == 2
        finally:
            os.unlink(temp_path)


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_import_load_okx_data(self):
        """Test load_okx_data can be imported."""
        from src.data.ccxt_loader import load_okx_data
        assert load_okx_data is not None
    
    def test_import_load_hybrid_data(self):
        """Test load_hybrid_data can be imported."""
        from src.data.ccxt_loader import load_hybrid_data
        assert load_hybrid_data is not None


class TestDataModuleExports:
    """Tests for data module exports."""
    
    def test_import_from_data_module(self):
        """Test all new exports from data module."""
        from src.data import (
            CCXTLoader,
            KaggleOKXHybridLoader,
            OKXConfig,
            load_hybrid_data,
            load_okx_data,
        )
        
        assert CCXTLoader is not None
        assert KaggleOKXHybridLoader is not None
        assert OKXConfig is not None
        assert load_okx_data is not None
        assert load_hybrid_data is not None
