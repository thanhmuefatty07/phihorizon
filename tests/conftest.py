"""
Test fixtures and configuration for PhiHorizon.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=500, freq='1h')
    np.random.seed(42)
    
    close = 40000 + np.cumsum(np.random.randn(500) * 100)
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': close - np.random.rand(500) * 50,
        'high': close + np.random.rand(500) * 100,
        'low': close - np.random.rand(500) * 100,
        'close': close,
        'volume': np.random.rand(500) * 1000000
    }).set_index('timestamp')


@pytest.fixture
def sample_returns():
    """Generate sample returns for testing."""
    np.random.seed(42)
    return pd.Series(np.random.randn(252) * 0.02)

