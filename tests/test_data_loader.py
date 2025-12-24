"""
Unit Tests for Hybrid Data Loader

Tests data loading, quality validation, and drift detection.
"""

import numpy as np
import pandas as pd
import pytest


class TestDataEnums:
    """Tests for data enums."""
    
    def test_import_enums(self):
        """Test enums can be imported."""
        from src.data import DataSource, DataPurpose
        
        assert DataSource.KAGGLE.value == "kaggle"
        assert DataSource.YFINANCE.value == "yfinance"
        assert DataPurpose.TRAINING.value == "training"
        assert DataPurpose.VALIDATION.value == "validation"


class TestDataConfig:
    """Tests for DataConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from src.data import DataConfig
        
        config = DataConfig()
        assert config.symbol == "BTC-USD"
        assert config.holdout_ratio == 0.15
        assert config.min_rows == 1000
    
    def test_custom_config(self):
        """Test custom configuration."""
        from src.data import DataConfig, DataPurpose
        
        config = DataConfig(
            symbol="ETH-USD",
            purpose=DataPurpose.VALIDATION,
            holdout_ratio=0.2,
        )
        assert config.symbol == "ETH-USD"
        assert config.holdout_ratio == 0.2


class TestHybridDataLoader:
    """Tests for HybridDataLoader."""
    
    def test_import(self):
        """Test loader can be imported."""
        from src.data import HybridDataLoader
        assert HybridDataLoader is not None
    
    def test_create_loader(self):
        """Test creating loader instance."""
        from src.data import HybridDataLoader, DataConfig
        
        config = DataConfig()
        loader = HybridDataLoader(config)
        assert loader is not None
    
    def test_source_selection_training(self):
        """Training should use Kaggle."""
        from src.data import HybridDataLoader, DataConfig, DataPurpose, DataSource
        
        config = DataConfig(purpose=DataPurpose.TRAINING)
        loader = HybridDataLoader(config)
        
        source = loader._select_source_for_purpose(DataPurpose.TRAINING)
        assert source == DataSource.KAGGLE
    
    def test_source_selection_validation(self):
        """Validation should use YFinance."""
        from src.data import HybridDataLoader, DataConfig, DataPurpose, DataSource
        
        config = DataConfig(purpose=DataPurpose.VALIDATION)
        loader = HybridDataLoader(config)
        
        source = loader._select_source_for_purpose(DataPurpose.VALIDATION)
        assert source == DataSource.YFINANCE
    
    def test_source_selection_ablation(self):
        """Ablation should use Kaggle (reproducible)."""
        from src.data import HybridDataLoader, DataConfig, DataPurpose, DataSource
        
        config = DataConfig()
        loader = HybridDataLoader(config)
        
        source = loader._select_source_for_purpose(DataPurpose.ABLATION)
        assert source == DataSource.KAGGLE
    
    def test_generate_synthetic(self):
        """Test synthetic data generation."""
        from src.data import HybridDataLoader, DataConfig
        
        loader = HybridDataLoader(DataConfig())
        df = loader._generate_synthetic()
        
        assert len(df) == 15000
        assert 'open' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns
    
    def test_standardize_columns(self):
        """Test column standardization."""
        from src.data import HybridDataLoader, DataConfig
        
        loader = HybridDataLoader(DataConfig())
        
        # Input with non-standard column names
        df = pd.DataFrame({
            'Open': [100, 101],
            'HIGH': [102, 103],
            'Low': [98, 99],
            'CLOSE': [101, 102],
            'Volume': [1000, 1100],
        })
        
        result = loader._standardize_columns(df)
        
        assert 'open' in result.columns
        assert 'high' in result.columns
        assert 'close' in result.columns


class TestDataQualityReport:
    """Tests for DataQualityReport."""
    
    def test_quality_report_creation(self):
        """Test quality report creation."""
        from src.data.hybrid_loader import DataQualityReport, DataSource
        
        report = DataQualityReport(
            source=DataSource.YFINANCE,
            rows=10000,
            columns=6,
            missing_pct=0.01,
            date_range=(None, None),
            has_gaps=False,
            gap_count=0,
            quality_score=0.9,
            warnings=[],
        )
        
        assert report.is_acceptable(min_score=0.7)
        assert not report.is_acceptable(min_score=0.95)
    
    def test_quality_validation(self):
        """Test data quality validation."""
        from src.data import HybridDataLoader, DataConfig, DataSource
        
        loader = HybridDataLoader(DataConfig())
        df = loader._generate_synthetic()
        
        report = loader._validate_data(df, DataSource.SYNTHETIC)
        
        assert report.rows == 15000
        assert 0 <= report.quality_score <= 1
        assert "SYNTHETIC" in report.warnings[0]


class TestHoldoutSplit:
    """Tests for holdout splitting."""
    
    def test_split_ratio(self):
        """Test holdout split ratio."""
        from src.data import HybridDataLoader, DataConfig
        
        loader = HybridDataLoader(DataConfig(holdout_ratio=0.2))
        
        df = pd.DataFrame({'a': range(100)})
        train, holdout = loader.split_holdout(df)
        
        assert len(train) == 80
        assert len(holdout) == 20


class TestDriftDetection:
    """Tests for concept drift detection."""
    
    def test_no_drift(self):
        """Test drift detection when no drift."""
        from src.data import HybridDataLoader, DataConfig
        
        loader = HybridDataLoader(DataConfig())
        
        # Same distribution
        np.random.seed(42)
        ref = pd.DataFrame({'close': np.random.randn(100) + 100})
        cur = pd.DataFrame({'close': np.random.randn(100) + 100})
        
        drift = loader.detect_drift(ref, cur, ['close'])
        
        # Should be low drift
        assert drift['close'] < 1.0
    
    def test_significant_drift(self):
        """Test drift detection when significant drift."""
        from src.data import HybridDataLoader, DataConfig
        
        loader = HybridDataLoader(DataConfig())
        
        # Different distributions
        ref = pd.DataFrame({'close': np.random.randn(100) + 100})
        cur = pd.DataFrame({'close': np.random.randn(100) + 200})  # Shifted
        
        drift = loader.detect_drift(ref, cur, ['close'])
        
        # Should detect significant drift
        assert drift['close'] > 5.0  # >5 std deviation shift


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_import_convenience_functions(self):
        """Test convenience functions can be imported."""
        from src.data import load_data_for_research, load_data_for_validation
        
        assert load_data_for_research is not None
        assert load_data_for_validation is not None
