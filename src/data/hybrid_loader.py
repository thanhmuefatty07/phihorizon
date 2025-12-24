#!/usr/bin/env python3
"""
PhiHorizon - Hybrid Data Pipeline

Research-backed data management with:
1. Kaggle datasets for training/backtesting (reproducible)
2. YFinance for real-time validation (current market conditions)
3. Automatic source selection based on use case
4. Data quality validation and drift detection

Usage Strategy:
- RESEARCH: Use Kaggle (fixed data, reproducible results)
- VALIDATION: Use YFinance (fresh data, generalization test)
- PRODUCTION: Kaggle retrain + YFinance predict
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Data source types."""
    KAGGLE = "kaggle"
    YFINANCE = "yfinance"
    SYNTHETIC = "synthetic"
    CACHED = "cached"


class DataPurpose(Enum):
    """Purpose determines which data source to use."""
    TRAINING = "training"          # Use Kaggle (stable, large)
    BACKTESTING = "backtesting"    # Use Kaggle (reproducible)
    VALIDATION = "validation"      # Use YFinance (fresh)
    PRODUCTION = "production"      # Use YFinance (real-time)
    ABLATION = "ablation"          # Use Kaggle (reproducible comparison)


@dataclass
class DataConfig:
    """Configuration for data loading."""
    symbol: str = "BTC-USD"
    purpose: DataPurpose = DataPurpose.TRAINING
    
    # Kaggle settings
    kaggle_dataset: str = "mczielinski/bitcoin-historical-data"
    kaggle_file: str = "bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv"
    
    # YFinance settings
    yf_period: str = "2y"
    yf_interval: str = "1h"
    
    # Data quality
    min_rows: int = 1000
    max_missing_pct: float = 0.05
    
    # Holdout split
    holdout_ratio: float = 0.15
    
    # Cache
    cache_dir: str = ".cache/data"
    cache_hours: int = 24


@dataclass
class DataQualityReport:
    """Report on data quality."""
    source: DataSource
    rows: int
    columns: int
    missing_pct: float
    date_range: Tuple[datetime, datetime]
    has_gaps: bool
    gap_count: int
    quality_score: float  # 0-1
    warnings: List[str]
    
    def is_acceptable(self, min_score: float = 0.7) -> bool:
        return self.quality_score >= min_score


class HybridDataLoader:
    """
    Hybrid data loader that selects source based on purpose.
    
    Research Phase: Kaggle (reproducible, large dataset)
    Validation Phase: YFinance (fresh, real market conditions)
    Production: YFinance real-time + Kaggle periodic retrain
    """
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self._setup_cache()
    
    def _setup_cache(self):
        """Setup cache directory."""
        cache_path = Path(self.config.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
    
    def load(
        self,
        purpose: DataPurpose = None,
        force_source: DataSource = None,
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Load data based on purpose or forced source.
        
        Args:
            purpose: DataPurpose to determine source automatically
            force_source: Override automatic source selection
            
        Returns:
            (DataFrame with OHLCV, DataQualityReport)
        """
        purpose = purpose or self.config.purpose
        
        # Determine source
        if force_source:
            source = force_source
        else:
            source = self._select_source_for_purpose(purpose)
        
        logger.info(f"Loading data for {purpose.value} from {source.value}")
        
        # Load data
        if source == DataSource.KAGGLE:
            df = self._load_kaggle()
        elif source == DataSource.YFINANCE:
            df = self._load_yfinance()
        elif source == DataSource.CACHED:
            df = self._load_cached()
        else:
            df = self._generate_synthetic()
        
        # Fallback chain
        if df is None or len(df) < self.config.min_rows:
            logger.warning(f"{source.value} failed, trying fallback...")
            df = self._load_with_fallback()
        
        # Validate and generate report
        report = self._validate_data(df, source)
        
        # Cache if from API
        if source == DataSource.YFINANCE:
            self._cache_data(df)
        
        return df, report
    
    def _select_source_for_purpose(self, purpose: DataPurpose) -> DataSource:
        """Select appropriate data source for purpose."""
        mapping = {
            DataPurpose.TRAINING: DataSource.KAGGLE,
            DataPurpose.BACKTESTING: DataSource.KAGGLE,
            DataPurpose.VALIDATION: DataSource.YFINANCE,
            DataPurpose.PRODUCTION: DataSource.YFINANCE,
            DataPurpose.ABLATION: DataSource.KAGGLE,
        }
        return mapping.get(purpose, DataSource.KAGGLE)
    
    def _load_kaggle(self) -> Optional[pd.DataFrame]:
        """Load data from Kaggle dataset."""
        try:
            # Check if running on Kaggle
            if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
                # Try to find BTC data in Kaggle inputs
                import glob
                csv_files = glob.glob('/kaggle/input/**/*.csv', recursive=True)
                btc_files = [f for f in csv_files if 'btc' in f.lower() or 'bitcoin' in f.lower()]
                
                if btc_files:
                    df = pd.read_csv(btc_files[0])
                    return self._standardize_columns(df)
            
            # Local Kaggle cache
            cache_file = Path(self.config.cache_dir) / "kaggle_btc.csv"
            if cache_file.exists():
                df = pd.read_csv(cache_file)
                return self._standardize_columns(df)
            
            logger.warning("Kaggle data not found locally")
            return None
            
        except Exception as e:
            logger.error(f"Kaggle load error: {e}")
            return None
    
    def _load_yfinance(self) -> Optional[pd.DataFrame]:
        """Load real-time data from YFinance."""
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(self.config.symbol)
            df = ticker.history(
                period=self.config.yf_period,
                interval=self.config.yf_interval
            )
            
            if len(df) < 100:
                return None
            
            df = df.reset_index()
            return self._standardize_columns(df)
            
        except ImportError:
            logger.warning("YFinance not installed")
            return None
        except Exception as e:
            logger.error(f"YFinance error: {e}")
            return None
    
    def _load_cached(self) -> Optional[pd.DataFrame]:
        """Load from cache if recent enough."""
        cache_file = Path(self.config.cache_dir) / "cached_data.parquet"
        
        if not cache_file.exists():
            return None
        
        # Check cache age
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        
        if age_hours > self.config.cache_hours:
            logger.info("Cache expired")
            return None
        
        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            logger.error(f"Cache read error: {e}")
            return None
    
    def _cache_data(self, df: pd.DataFrame):
        """Cache data for later use."""
        cache_file = Path(self.config.cache_dir) / "cached_data.parquet"
        try:
            df.to_parquet(cache_file, index=False)
            logger.info(f"Cached {len(df)} rows")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def _generate_synthetic(self) -> pd.DataFrame:
        """Generate synthetic data for testing."""
        logger.warning("Generating SYNTHETIC data - NOT for production!")
        
        np.random.seed(42)
        n = 15000
        
        # Random walk price
        price = 30000 + np.cumsum(np.random.randn(n) * 50)
        
        df = pd.DataFrame({
            'timestamp': pd.date_range(end=pd.Timestamp.now(), periods=n, freq='H'),
            'open': price,
            'high': price * 1.002,
            'low': price * 0.998,
            'close': price + np.random.randn(n) * 20,
            'volume': np.random.rand(n) * 1000000,
        })
        
        return df
    
    def _load_with_fallback(self) -> pd.DataFrame:
        """Try loading from multiple sources with fallback."""
        sources = [DataSource.YFINANCE, DataSource.CACHED, DataSource.SYNTHETIC]
        
        for source in sources:
            logger.info(f"Trying fallback source: {source.value}")
            
            if source == DataSource.YFINANCE:
                df = self._load_yfinance()
            elif source == DataSource.CACHED:
                df = self._load_cached()
            else:
                df = self._generate_synthetic()
            
            if df is not None and len(df) >= self.config.min_rows:
                return df
        
        # Last resort: synthetic
        return self._generate_synthetic()
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and types."""
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Rename common variants
        rename_map = {
            'datetime': 'timestamp',
            'date': 'timestamp',
            'time': 'timestamp',
            'price': 'close',
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        # Ensure required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                if col == 'volume':
                    df[col] = 0
                else:
                    df[col] = df.get('close', 0)
        
        # Convert types
        for col in required:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        return df.dropna(subset=required)
    
    def _validate_data(self, df: pd.DataFrame, source: DataSource) -> DataQualityReport:
        """Validate data quality and generate report."""
        warnings = []
        
        # Basic stats
        rows = len(df)
        cols = len(df.columns)
        missing_pct = df.isnull().sum().sum() / (rows * cols)
        
        # Date range
        if 'timestamp' in df.columns:
            date_range = (df['timestamp'].min(), df['timestamp'].max())
        else:
            date_range = (None, None)
        
        # Gap detection
        has_gaps = False
        gap_count = 0
        if 'timestamp' in df.columns and len(df) > 1:
            time_diffs = df['timestamp'].diff().dropna()
            median_diff = time_diffs.median()
            large_gaps = time_diffs > median_diff * 3
            has_gaps = large_gaps.any()
            gap_count = large_gaps.sum()
            
            if has_gaps:
                warnings.append(f"Data has {gap_count} gaps")
        
        # Quality score calculation
        quality_score = 1.0
        
        # Penalize missing data
        quality_score -= missing_pct * 2
        
        # Penalize gaps
        if has_gaps:
            quality_score -= min(0.2, gap_count / 100)
        
        # Penalize small datasets
        if rows < 5000:
            quality_score -= 0.1
        if rows < 1000:
            quality_score -= 0.2
            warnings.append("Dataset is small (<1000 rows)")
        
        # Penalize synthetic data
        if source == DataSource.SYNTHETIC:
            quality_score -= 0.3
            warnings.append("Using SYNTHETIC data - not suitable for production")
        
        quality_score = max(0, min(1, quality_score))
        
        return DataQualityReport(
            source=source,
            rows=rows,
            columns=cols,
            missing_pct=missing_pct,
            date_range=date_range,
            has_gaps=has_gaps,
            gap_count=gap_count,
            quality_score=quality_score,
            warnings=warnings,
        )
    
    def split_holdout(
        self,
        df: pd.DataFrame,
        holdout_ratio: float = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and holdout sets.
        
        Returns:
            (training_df, holdout_df)
        """
        ratio = holdout_ratio or self.config.holdout_ratio
        split_idx = int(len(df) * (1 - ratio))
        
        train_df = df.iloc[:split_idx].copy()
        holdout_df = df.iloc[split_idx:].copy()
        
        logger.info(f"Split: {len(train_df)} train, {len(holdout_df)} holdout")
        
        return train_df, holdout_df
    
    def detect_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_cols: List[str] = None,
    ) -> Dict[str, float]:
        """
        Detect concept drift between reference and current data.
        
        Uses simple statistical tests. Returns dict of drift scores per feature.
        Higher score = more drift.
        """
        if feature_cols is None:
            feature_cols = ['close', 'volume']
        
        drift_scores = {}
        
        for col in feature_cols:
            if col not in reference_df.columns or col not in current_df.columns:
                continue
            
            ref_values = reference_df[col].dropna().values
            cur_values = current_df[col].dropna().values
            
            if len(ref_values) < 30 or len(cur_values) < 30:
                drift_scores[col] = 0.0
                continue
            
            # Normalized mean difference
            ref_mean = np.mean(ref_values)
            cur_mean = np.mean(cur_values)
            ref_std = np.std(ref_values)
            
            if ref_std > 0:
                drift = abs(cur_mean - ref_mean) / ref_std
            else:
                drift = 0.0
            
            drift_scores[col] = float(drift)
        
        return drift_scores


def load_data_for_research(symbol: str = "BTC-USD") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function for research: loads Kaggle data with holdout split.
    
    Returns:
        (training_df, holdout_df)
    """
    config = DataConfig(symbol=symbol, purpose=DataPurpose.TRAINING)
    loader = HybridDataLoader(config)
    
    df, report = loader.load()
    logger.info(f"Loaded {report.rows} rows, quality: {report.quality_score:.2f}")
    
    if report.warnings:
        for w in report.warnings:
            logger.warning(w)
    
    return loader.split_holdout(df)


def load_data_for_validation(symbol: str = "BTC-USD") -> pd.DataFrame:
    """
    Convenience function for validation: loads fresh YFinance data.
    
    Returns:
        Fresh market data DataFrame
    """
    config = DataConfig(symbol=symbol, purpose=DataPurpose.VALIDATION)
    loader = HybridDataLoader(config)
    
    df, report = loader.load()
    logger.info(f"Validation data: {report.rows} rows from {report.source.value}")
    
    return df


__all__ = [
    'DataSource',
    'DataPurpose',
    'DataConfig',
    'DataQualityReport',
    'HybridDataLoader',
    'load_data_for_research',
    'load_data_for_validation',
]
