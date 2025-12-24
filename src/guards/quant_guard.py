#!/usr/bin/env python3
"""
PhiHorizon V7.0 ML Guard 1: Quant Data Quality

Quality control for numerical/quantitative data before feeding to CORE 1.

Features:
- Anomaly detection (outliers)
- Missing data handling
- Data freshness validation
- Noise filtering
- Range validation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class QuantGuardConfig:
    """Configuration for Quant Data Quality Guard."""
    
    # Anomaly detection
    zscore_threshold: float = 3.0  # Z-score for outlier detection
    iqr_multiplier: float = 1.5   # IQR multiplier for outliers
    
    # Missing data thresholds
    max_missing_pct: float = 0.10  # Max 10% missing allowed
    
    # Freshness
    max_data_age_hours: int = 24   # Data must be < 24 hours old
    
    # Feature ranges (expected min/max for validation)
    feature_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "open": (0, 1_000_000),
        "high": (0, 1_000_000),
        "low": (0, 1_000_000),
        "close": (0, 1_000_000),
        "volume": (0, 1e15),
        "funding_rate": (-0.01, 0.01),
        "open_interest": (0, 1e12),
        "long_short_ratio": (0.1, 10.0),
        "fear_greed": (0, 100),
        "btc_dominance": (20, 80),
        "hash_rate": (0, 1e21),
        "whale_netflow": (-1.0, 1.0),
    })
    
    # Quality score thresholds
    min_quality_score: float = 0.7  # Minimum acceptable quality


# ============================================================
# QUANT GUARD
# ============================================================

class QuantGuard:
    """
    ML Guard 1: Quality control for quantitative data.
    
    Responsibilities:
    1. Detect and handle anomalies/outliers
    2. Validate data freshness
    3. Handle missing values
    4. Ensure data is within expected ranges
    5. Calculate overall quality score
    
    Used to filter/clean data before CORE 1 (Quant Transformer).
    """
    
    def __init__(self, config: Optional[QuantGuardConfig] = None):
        """Initialize the Quant Guard."""
        self.config = config or QuantGuardConfig()
        self._quality_history: List[float] = []
        
    # ========================================================
    # ANOMALY DETECTION
    # ========================================================
    
    def detect_outliers_zscore(
        self,
        data: pd.Series,
        threshold: Optional[float] = None
    ) -> pd.Series:
        """
        Detect outliers using Z-score method.
        
        Args:
            data: Numeric series
            threshold: Z-score threshold (default from config)
            
        Returns:
            Boolean series (True = outlier)
        """
        threshold = threshold or self.config.zscore_threshold
        
        if data.std() == 0:
            return pd.Series([False] * len(data), index=data.index)
            
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
    
    def detect_outliers_iqr(
        self,
        data: pd.Series,
        multiplier: Optional[float] = None
    ) -> pd.Series:
        """
        Detect outliers using IQR method.
        
        Args:
            data: Numeric series
            multiplier: IQR multiplier (default from config)
            
        Returns:
            Boolean series (True = outlier)
        """
        multiplier = multiplier or self.config.iqr_multiplier
        
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        
        return (data < lower) | (data > upper)
    
    def handle_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "clip"
    ) -> pd.DataFrame:
        """
        Handle outliers in DataFrame.
        
        Args:
            df: Input DataFrame
            columns: Columns to check (default: all numeric)
            method: "clip" (cap at bounds), "nan" (set to NaN), "median" (replace with median)
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
        for col in columns:
            if col not in df.columns:
                continue
                
            outliers = self.detect_outliers_iqr(df[col])
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                logger.info(f"Found {outlier_count} outliers in {col}")
                
                if method == "clip":
                    q1, q3 = df[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    df[col] = df[col].clip(
                        lower=q1 - self.config.iqr_multiplier * iqr,
                        upper=q3 + self.config.iqr_multiplier * iqr
                    )
                elif method == "nan":
                    df.loc[outliers, col] = np.nan
                elif method == "median":
                    df.loc[outliers, col] = df[col].median()
                    
        return df
    
    # ========================================================
    # MISSING DATA HANDLING
    # ========================================================
    
    def check_missing(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Check missing data percentage per column.
        
        Returns:
            Dict of column: missing_pct
        """
        return (df.isnull().sum() / len(df)).to_dict()
    
    def has_acceptable_missing(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Check if missing data is within acceptable range.
        
        Returns:
            Tuple of (is_acceptable, details)
        """
        missing = self.check_missing(df)
        
        problematic = {
            k: v for k, v in missing.items()
            if v > self.config.max_missing_pct
        }
        
        return len(problematic) == 0, {
            "missing_by_column": missing,
            "problematic_columns": problematic,
            "max_allowed": self.config.max_missing_pct
        }
    
    def impute_missing(
        self,
        df: pd.DataFrame,
        method: str = "ffill",
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Impute missing values.
        
        Args:
            df: Input DataFrame
            method: "ffill", "bfill", "median", "mean", "zero"
            columns: Columns to impute (default: all)
            
        Returns:
            Imputed DataFrame
        """
        df = df.copy()
        
        if columns is None:
            columns = df.columns.tolist()
            
        for col in columns:
            if col not in df.columns:
                continue
                
            if df[col].isnull().sum() == 0:
                continue
                
            if method == "ffill":
                df[col] = df[col].fillna(method="ffill")
            elif method == "bfill":
                df[col] = df[col].fillna(method="bfill")
            elif method == "median":
                df[col] = df[col].fillna(df[col].median())
            elif method == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif method == "zero":
                df[col] = df[col].fillna(0)
                
        return df
    
    # ========================================================
    # DATA FRESHNESS
    # ========================================================
    
    def check_freshness(
        self,
        last_timestamp: datetime,
        max_age_hours: Optional[int] = None
    ) -> Tuple[bool, float]:
        """
        Check if data is fresh enough.
        
        Args:
            last_timestamp: Most recent data timestamp
            max_age_hours: Maximum allowed age
            
        Returns:
            Tuple of (is_fresh, age_in_hours)
        """
        max_age = max_age_hours or self.config.max_data_age_hours
        
        now = datetime.now()
        if hasattr(last_timestamp, 'to_pydatetime'):
            last_timestamp = last_timestamp.to_pydatetime()
            
        age = (now - last_timestamp).total_seconds() / 3600
        
        return age <= max_age, age
    
    # ========================================================
    # RANGE VALIDATION
    # ========================================================
    
    def validate_ranges(
        self,
        df: pd.DataFrame,
        ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, Dict]:
        """
        Validate that data is within expected ranges.
        
        Returns:
            Dict with validation results per column
        """
        ranges = ranges or self.config.feature_ranges
        results = {}
        
        for col, (min_val, max_val) in ranges.items():
            if col not in df.columns:
                continue
                
            out_of_range = (df[col] < min_val) | (df[col] > max_val)
            out_count = out_of_range.sum()
            
            results[col] = {
                "expected_range": (min_val, max_val),
                "actual_range": (df[col].min(), df[col].max()),
                "out_of_range_count": out_count,
                "out_of_range_pct": out_count / len(df) if len(df) > 0 else 0,
                "is_valid": out_count == 0
            }
            
        return results
    
    # ========================================================
    # QUALITY SCORING
    # ========================================================
    
    def calculate_quality_score(
        self,
        df: pd.DataFrame,
        timestamp_col: Optional[str] = "timestamp"
    ) -> Dict[str, Any]:
        """
        Calculate overall data quality score.
        
        Components:
        1. Completeness (missing data)
        2. Validity (range checks)
        3. Freshness (data age)
        4. Consistency (outliers)
        
        Returns:
            Dict with scores and overall quality
        """
        scores = {}
        
        # 1. Completeness
        missing = self.check_missing(df)
        avg_missing = sum(missing.values()) / len(missing) if missing else 0
        scores["completeness"] = 1 - avg_missing
        
        # 2. Validity
        range_results = self.validate_ranges(df)
        if range_results:
            valid_pcts = [
                1 - r["out_of_range_pct"]
                for r in range_results.values()
            ]
            scores["validity"] = sum(valid_pcts) / len(valid_pcts)
        else:
            scores["validity"] = 1.0
            
        # 3. Freshness
        if timestamp_col and timestamp_col in df.columns:
            last_ts = pd.to_datetime(df[timestamp_col]).max()
            is_fresh, age = self.check_freshness(last_ts)
            # Score decays with age
            scores["freshness"] = max(0, 1 - (age / self.config.max_data_age_hours))
        else:
            scores["freshness"] = 1.0
            
        # 4. Consistency (outlier ratio)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_ratios = []
        for col in numeric_cols:
            outliers = self.detect_outliers_zscore(df[col])
            outlier_ratios.append(outliers.mean())
        scores["consistency"] = 1 - (sum(outlier_ratios) / len(outlier_ratios)) if outlier_ratios else 1.0
        
        # Overall score (weighted average)
        weights = {
            "completeness": 0.3,
            "validity": 0.25,
            "freshness": 0.25,
            "consistency": 0.2
        }
        
        overall = sum(scores[k] * weights[k] for k in weights)
        
        result = {
            "scores": scores,
            "overall": overall,
            "is_acceptable": overall >= self.config.min_quality_score,
            "min_threshold": self.config.min_quality_score,
            "timestamp": datetime.now().isoformat()
        }
        
        # Track history
        self._quality_history.append(overall)
        
        return result
    
    # ========================================================
    # FULL PIPELINE
    # ========================================================
    
    def process(
        self,
        df: pd.DataFrame,
        timestamp_col: Optional[str] = "timestamp"
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Full quality control pipeline.
        
        Steps:
        1. Check quality score
        2. Handle outliers
        3. Impute missing values
        4. Validate output
        
        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column
            
        Returns:
            Tuple of (cleaned_df, quality_report)
        """
        logger.info(f"Processing {len(df)} rows through Quant Guard")
        
        # Initial quality check
        initial_quality = self.calculate_quality_score(df, timestamp_col)
        
        # Handle outliers
        df_cleaned = self.handle_outliers(df, method="clip")
        
        # Impute missing
        df_cleaned = self.impute_missing(df_cleaned, method="ffill")
        df_cleaned = self.impute_missing(df_cleaned, method="bfill")  # For remaining
        
        # Final quality check
        final_quality = self.calculate_quality_score(df_cleaned, timestamp_col)
        
        report = {
            "initial_quality": initial_quality,
            "final_quality": final_quality,
            "rows_processed": len(df),
            "improvement": final_quality["overall"] - initial_quality["overall"],
            "passed": final_quality["is_acceptable"]
        }
        
        if not report["passed"]:
            logger.warning(
                f"Data quality below threshold: {final_quality['overall']:.2f} < "
                f"{self.config.min_quality_score}"
            )
            
        return df_cleaned, report


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def validate_quant_data(df: pd.DataFrame) -> Tuple[bool, Dict]:
    """Quick validation of quantitative data."""
    guard = QuantGuard()
    _, report = guard.process(df)
    return report["passed"], report


def clean_quant_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean quantitative data using default settings."""
    guard = QuantGuard()
    cleaned, _ = guard.process(df)
    return cleaned


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create test data
    np.random.seed(42)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="D"),
        "close": np.random.uniform(40000, 50000, 100),
        "volume": np.random.uniform(1e9, 5e9, 100),
        "funding_rate": np.random.uniform(-0.001, 0.001, 100),
        "fear_greed": np.random.randint(20, 80, 100)
    })
    
    # Add some noise
    df.loc[5, "close"] = 1_000_000  # Outlier
    df.loc[10, "volume"] = np.nan   # Missing
    df.loc[15, "funding_rate"] = 0.5  # Out of range
    
    guard = QuantGuard()
    cleaned, report = guard.process(df)
    
    print("\n=== Quality Report ===")
    print(f"Initial Quality: {report['initial_quality']['overall']:.2f}")
    print(f"Final Quality: {report['final_quality']['overall']:.2f}")
    print(f"Improvement: {report['improvement']:.2f}")
    print(f"Passed: {report['passed']}")
