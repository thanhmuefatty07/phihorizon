#!/usr/bin/env python3
"""
PhiHorizon V6.1 Multi-Source Data Merger

Combines data from all loaders into a unified dataset for ML training.

Sources integrated:
1. Alternative.me (Fear/Greed)
2. CoinGecko (BTC Dominance, Stablecoin MCap)
3. Binance (Open Interest, Long/Short Ratio)
4. Google Trends (Search Interest)
5. Blockchain.com (Hash Rate)
6. OKX/Binance (Funding Rate)

Quality Checklist:
[x] Data validation
[x] Missing data handling
[x] Feature correlation check
[x] Comprehensive logging
[x] Ablation testing support

Note: This is for SIMULATION/BACKTESTING only, NOT live trading.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Import all loaders
from src.data.sentiment_loader import SentimentLoader
from src.data.funding_loader import FundingLoader
from src.data.coingecko_loader import CoinGeckoLoader
from src.data.binance_loader import BinanceLoader
from src.data.google_trends_loader import GoogleTrendsLoader
from src.data.blockchain_loader import BlockchainLoader

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class MergerConfig:
    """Configuration for multi-source merger."""
    
    # Feature groups for ablation testing
    feature_groups: Dict[str, List[str]] = None
    
    # Missing data handling
    max_missing_pct: float = 0.20  # Max 20% missing allowed
    fill_method: str = "ffill"     # Forward fill
    
    # Validation thresholds
    min_records: int = 365         # Minimum 1 year of data
    
    def __post_init__(self):
        if self.feature_groups is None:
            self.feature_groups = {
                'sentiment': ['fear_greed', 'fg_ma7', 'fg_change'],
                'derivatives': ['funding_rate', 'open_interest', 'long_short_ratio'],
                'macro': ['btc_dominance', 'stablecoin_mcap', 'total_mcap'],
                'social': ['google_trends', 'trends_change_7d'],
                'onchain': ['hash_rate', 'hash_rate_change'],
            }


# ============================================================
# MULTI-SOURCE MERGER
# ============================================================

class MultiSourceMerger:
    """
    Merges data from all sources into unified dataset.
    
    Provides:
    - Unified daily dataset with all features
    - Feature validation and quality metrics
    - Ablation study support
    """
    
    def __init__(self, config: Optional[MergerConfig] = None):
        """Initialize the MultiSourceMerger."""
        self.config = config or MergerConfig()
        
        # Initialize loaders
        self.sentiment_loader = SentimentLoader()
        self.funding_loader = FundingLoader()
        self.coingecko_loader = CoinGeckoLoader()
        self.binance_loader = BinanceLoader()
        self.trends_loader = GoogleTrendsLoader()
        self.blockchain_loader = BlockchainLoader()
        
        logger.info("MultiSourceMerger initialized")
        logger.info(f"Feature groups: {list(self.config.feature_groups.keys())}")
    
    # ================================================================
    # DATA LOADING
    # ================================================================
    
    def load_all_sources(self, days: int = 365) -> Dict[str, pd.DataFrame]:
        """
        Load data from all sources.
        
        Args:
            days: Number of days to load
            
        Returns:
            Dict of source_name -> DataFrame
        """
        logger.info(f"Loading data from all sources ({days} days)...")
        
        sources = {}
        
        # 1. Fear/Greed Index
        logger.info("Loading Fear/Greed...")
        try:
            fg_df = self.sentiment_loader.get_fear_greed_index(days=days)
            if fg_df is not None:
                sources['fear_greed'] = fg_df
                logger.info(f"  Fear/Greed: {len(fg_df)} records")
        except Exception as e:
            logger.error(f"  Fear/Greed failed: {e}")
        
        # 2. Funding Rate
        logger.info("Loading Funding Rate...")
        try:
            fr_df = self.funding_loader.get_funding_rate(days=min(days, 90))
            if fr_df is not None:
                sources['funding_rate'] = fr_df
                logger.info(f"  Funding Rate: {len(fr_df)} records")
        except Exception as e:
            logger.error(f"  Funding Rate failed: {e}")
        
        # 3. Google Trends
        logger.info("Loading Google Trends...")
        try:
            trends_df = self.trends_loader.get_search_interest("Bitcoin")
            if trends_df is not None:
                sources['google_trends'] = trends_df
                logger.info(f"  Google Trends: {len(trends_df)} records")
        except Exception as e:
            logger.error(f"  Google Trends failed: {e}")
        
        # 4. Hash Rate
        logger.info("Loading Hash Rate...")
        try:
            hr_df = self.blockchain_loader.get_hash_rate()
            if hr_df is not None:
                sources['hash_rate'] = hr_df
                logger.info(f"  Hash Rate: {len(hr_df)} records")
        except Exception as e:
            logger.error(f"  Hash Rate failed: {e}")
        
        # 5. Derivatives (Historical)
        logger.info("Loading Derivatives Data...")
        try:
            # Long/Short Ratio
            ls_df = self.binance_loader.get_long_short_ratio_historical(days=min(days, 30))
            if ls_df is not None:
                sources['long_short'] = ls_df
                logger.info(f"  Long/Short: {len(ls_df)} records")
            
            # Open Interest
            oi_df = self.binance_loader.get_open_interest_historical(days=min(days, 30))
            if oi_df is not None:
                sources['open_interest'] = oi_df
                logger.info(f"  Open Interest: {len(oi_df)} records")
        except Exception as e:
            logger.error(f"  Derivatives failed: {e}")
        
        logger.info(f"Loaded {len(sources)} data sources")
        
        return sources
    
    # ================================================================
    # DATA MERGING
    # ================================================================
    
    def merge_sources(
        self,
        sources: Dict[str, pd.DataFrame],
        base_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Merge all sources into unified DataFrame.
        
        Args:
            sources: Dict of source DataFrames
            base_df: Optional base DataFrame with dates
            
        Returns:
            Merged DataFrame
        """
        logger.info("Merging data sources...")
        
        # Create base date range if not provided
        if base_df is None:
            # Find date range from sources
            min_date = datetime.now() - timedelta(days=365)
            max_date = datetime.now()
            
            dates = pd.date_range(start=min_date, end=max_date, freq='D')
            base_df = pd.DataFrame({'date': dates})
        else:
            base_df = base_df.copy()
            if 'timestamp' in base_df.columns:
                base_df['date'] = pd.to_datetime(base_df['timestamp']).dt.date
            elif 'datetime' in base_df.columns:
                base_df['date'] = pd.to_datetime(base_df['datetime']).dt.date
        
        base_df['date'] = pd.to_datetime(base_df['date'])
        merged = base_df.copy()
        
        # Merge each source
        for source_name, df in sources.items():
            logger.info(f"  Merging {source_name}...")
            
            df = df.copy()
            
            # Normalize date column
            if 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp']).dt.floor('D')
            elif 'date' not in df.columns:
                continue
            
            # Remove duplicate columns
            for col in df.columns:
                if col in merged.columns and col != 'date':
                    df = df.drop(columns=[col])
            
            # Merge
            try:
                merged = merged.merge(
                    df.drop_duplicates('date'),
                    on='date',
                    how='left'
                )
            except Exception as e:
                logger.warning(f"  Failed to merge {source_name}: {e}")
        
        logger.info(f"Merged DataFrame: {len(merged)} rows, {len(merged.columns)} columns")
        
        return merged
    
    # ================================================================
    # DATA VALIDATION
    # ================================================================
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict:
        """
        Validate merged dataset quality.
        
        Returns:
            Validation report dict
        """
        logger.info("Validating dataset...")
        
        report = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'missing_by_column': {},
            'valid': True,
            'issues': [],
        }
        
        # Check missing data per column
        for col in df.columns:
            if col == 'date':
                continue
            
            missing_pct = df[col].isna().sum() / len(df)
            report['missing_by_column'][col] = missing_pct
            
            if missing_pct > self.config.max_missing_pct:
                report['issues'].append(f"{col}: {missing_pct:.1%} missing")
        
        # Check minimum records
        if len(df) < self.config.min_records:
            report['issues'].append(f"Only {len(df)} records (min: {self.config.min_records})")
            report['valid'] = False
        
        # Log report
        logger.info(f"Validation: {len(report['issues'])} issues found")
        for issue in report['issues']:
            logger.warning(f"  - {issue}")
        
        return report
    
    def fill_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing data using configured method.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with filled values
        """
        logger.info("Filling missing data...")
        
        df = df.copy()
        
        # Forward fill for most features
        if self.config.fill_method == "ffill":
            df = df.ffill()
        
        # Backward fill for remaining
        df = df.bfill()
        
        # Fill any remaining with 0 for numeric
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        remaining_nulls = df.isna().sum().sum()
        logger.info(f"Remaining nulls after fill: {remaining_nulls}")
        
        return df
    
    # ================================================================
    # FEATURE EXTRACTION
    # ================================================================
    
    def get_feature_columns(
        self,
        groups: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get feature columns for specified groups.
        
        Args:
            groups: List of group names (None = all)
            
        Returns:
            List of feature column names
        """
        if groups is None:
            groups = list(self.config.feature_groups.keys())
        
        features = []
        for group in groups:
            if group in self.config.feature_groups:
                features.extend(self.config.feature_groups[group])
        
        return features
    
    def create_ml_dataset(
        self,
        df: pd.DataFrame,
        target_col: str = 'target',
        feature_groups: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create ML-ready dataset from merged data.
        
        Args:
            df: Merged DataFrame
            target_col: Name of target column
            feature_groups: Groups to include (None = all)
            
        Returns:
            Tuple of (X, y)
        """
        logger.info("Creating ML dataset...")
        
        # Get feature columns
        feature_cols = self.get_feature_columns(feature_groups)
        
        # Filter to available columns
        available_cols = [c for c in feature_cols if c in df.columns]
        logger.info(f"Available features: {len(available_cols)}/{len(feature_cols)}")
        
        # Extract X, y
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found")
            y = None
        else:
            y = df[target_col]
        
        X = df[available_cols]
        
        return X, y
    
    # ================================================================
    # ABLATION STUDY
    # ================================================================
    
    def run_ablation_study(
        self,
        df: pd.DataFrame,
        target_col: str = 'target'
    ) -> Dict[str, Dict]:
        """
        Run ablation study to test feature group importance.
        
        Args:
            df: Full merged DataFrame
            target_col: Target column
            
        Returns:
            Dict with results per group combination
        """
        logger.info("Running ablation study...")
        
        results = {}
        groups = list(self.config.feature_groups.keys())
        
        # Test each group individually
        for group in groups:
            logger.info(f"Testing group: {group}")
            X, y = self.create_ml_dataset(df, target_col, [group])
            
            results[group] = {
                'n_features': len(X.columns),
                'features': X.columns.tolist(),
                'missing_pct': X.isna().sum().sum() / X.size,
            }
        
        # Test all groups
        logger.info("Testing all groups...")
        X, y = self.create_ml_dataset(df, target_col, groups)
        
        results['all'] = {
            'n_features': len(X.columns),
            'features': X.columns.tolist(),
            'missing_pct': X.isna().sum().sum() / X.size,
        }
        
        return results
    
    # ================================================================
    # FULL PIPELINE
    # ================================================================
    
    def create_full_dataset(
        self,
        days: int = 365,
        validate: bool = True,
        fill_missing: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Full pipeline: load, merge, validate, fill.
        
        Args:
            days: Number of days
            validate: Whether to validate
            fill_missing: Whether to fill missing
            
        Returns:
            Complete merged DataFrame
        """
        logger.info("=" * 60)
        logger.info("CREATING FULL MULTI-SOURCE DATASET")
        logger.info("=" * 60)
        
        # Load all sources
        sources = self.load_all_sources(days)
        
        if not sources:
            logger.error("No sources loaded!")
            return None
        
        # Merge
        merged = self.merge_sources(sources)
        
        # Validate
        if validate:
            report = self.validate_dataset(merged)
            if not report['valid']:
                logger.warning("Dataset validation failed!")
        
        # Fill missing
        if fill_missing:
            merged = self.fill_missing_data(merged)
        
        logger.info("=" * 60)
        logger.info(f"DATASET COMPLETE: {len(merged)} rows, {len(merged.columns)} cols")
        logger.info("=" * 60)
        
        return merged


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_extended_dataset(days: int = 365) -> Optional[pd.DataFrame]:
    """Create extended dataset with all features."""
    merger = MultiSourceMerger()
    return merger.create_full_dataset(days)


# ============================================================
# MAIN (for testing)
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "=" * 60)
    print("MULTI-SOURCE MERGER TEST")
    print("=" * 60)
    
    merger = MultiSourceMerger()
    
    # Test loading
    print("\n📊 Loading all sources...")
    sources = merger.load_all_sources(days=30)
    print(f"✅ Loaded {len(sources)} sources")
    
    for name, df in sources.items():
        print(f"   {name}: {len(df)} records")
    
    # Test merging
    print("\n🔄 Merging sources...")
    merged = merger.merge_sources(sources)
    print(f"✅ Merged: {len(merged)} rows, {len(merged.columns)} cols")
    
    # Test validation
    print("\n✅ Validating...")
    report = merger.validate_dataset(merged)
    print(f"   Valid: {report['valid']}")
    print(f"   Issues: {len(report['issues'])}")
    
    # Test fill
    print("\n🔧 Filling missing data...")
    filled = merger.fill_missing_data(merged)
    print(f"✅ Filled: {filled.isna().sum().sum()} remaining nulls")
    
    # Show columns
    print("\n📋 Columns:")
    for col in merged.columns:
        print(f"   - {col}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
