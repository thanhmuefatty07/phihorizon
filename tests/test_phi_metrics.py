"""
Phi Calculator and Consciousness Metrics Tests

Tests for IIT-based consciousness metrics.
"""

import pytest
import pandas as pd
import numpy as np


class TestPhiCalculator:
    """Tests for PhiCalculator class."""
    
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
        """Test that PhiCalculator can be imported."""
        from src.consciousness.metrics import PhiCalculator
        assert PhiCalculator is not None
    
    def test_init(self):
        """Test PhiCalculator initialization."""
        from src.consciousness.metrics import PhiCalculator
        
        calc = PhiCalculator(window_size=20)
        assert calc.window_size == 20
    
    def test_calculate_phi(self, sample_data):
        """Test Phi calculation returns valid value."""
        from src.consciousness.metrics import PhiCalculator
        
        calc = PhiCalculator(window_size=20)
        phi = calc.calculate_phi(sample_data)
        
        assert isinstance(phi, float)
        assert 0 <= phi <= 1, f"Phi {phi} should be between 0 and 1"
    
    def test_calculate_phi_insufficient_data(self):
        """Test Phi with insufficient data returns 0."""
        from src.consciousness.metrics import PhiCalculator
        
        calc = PhiCalculator(window_size=20)
        small_data = pd.DataFrame({
            'open': [100], 'high': [101], 'low': [99],
            'close': [100], 'volume': [1000]
        })
        
        phi = calc.calculate_phi(small_data)
        assert phi == 0.0


class TestIITCore:
    """Tests for IITCore class."""
    
    def test_import(self):
        """Test that IITCore can be imported."""
        from src.consciousness.metrics import IITCore
        assert IITCore is not None


class TestConsciousnessMetrics:
    """Tests for ConsciousnessMetrics dataclass."""
    
    def test_create_metrics(self):
        """Test creating consciousness metrics container."""
        from src.consciousness.metrics import ConsciousnessMetrics
        import time
        
        metrics = ConsciousnessMetrics(
            phi=0.7,
            valence=0.5,
            subjectivity=0.8,
            qualia=0.6,
            timestamp=time.time()
        )
        
        assert metrics.phi == 0.7
        assert metrics.valence == 0.5
    
    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        from src.consciousness.metrics import ConsciousnessMetrics
        import time
        
        metrics = ConsciousnessMetrics(
            phi=0.7, valence=0.5, subjectivity=0.8,
            qualia=0.6, timestamp=time.time()
        )
        
        d = metrics.to_dict()
        assert 'phi' in d
        assert d['phi'] == 0.7
