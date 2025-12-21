#!/usr/bin/env python3
"""
PhiHorizon - Data Utilities

Data manipulation, transformation, and analysis utilities.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Memory-optimized dtypes for financial data
OPTIMIZED_DTYPES = {
    "open": "float32",
    "high": "float32",
    "low": "float32",
    "close": "float32",
    "volume": "int32",  # Most exchanges use int32 for volume
    "timestamp": "datetime64[ns]",  # Keep full precision for timestamps
}

# Chunk sizes for memory-efficient processing
DEFAULT_CHUNK_SIZE = 50000  # Process 50K rows at a time
MAX_MEMORY_USAGE_MB = 100  # Target max memory usage


def optimize_dataframe_memory(df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by converting to optimal dtypes.

    Args:
        df: Input DataFrame
        copy: Whether to create a copy (avoid SettingWithCopyWarning)

    Returns:
        Memory-optimized DataFrame
    """
    if copy:
        df = df.copy()

    original_memory = df.memory_usage(deep=True).sum() / 1024**2

    # Convert to optimized dtypes
    for col in df.columns:
        if col in OPTIMIZED_DTYPES:
            try:
                if col == "timestamp":
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = pd.to_datetime(df[col])
                elif col == "volume":
                    # Convert to int, handling NaN values
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(OPTIMIZED_DTYPES[col])
                else:
                    df[col] = df[col].astype(OPTIMIZED_DTYPES[col])
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not optimize dtype for column {col}: {e}")

    optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
    memory_savings = original_memory - optimized_memory
    savings_pct = (memory_savings / original_memory) * 100 if original_memory > 0 else 0

    logger.info(".2f")
    return df


def chunk_dataframe(df: pd.DataFrame, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[pd.DataFrame]:
    """
    Split DataFrame into memory-efficient chunks.

    Args:
        df: Input DataFrame
        chunk_size: Number of rows per chunk

    Returns:
        List of DataFrame chunks
    """
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i : i + chunk_size].copy()
        chunks.append(chunk)

    logger.info(f"Split DataFrame into {len(chunks)} chunks of ~{chunk_size} rows each")
    return chunks


def get_memory_usage_mb(df: pd.DataFrame) -> float:
    """
    Get memory usage of DataFrame in MB.

    Args:
        df: Input DataFrame

    Returns:
        Memory usage in MB
    """
    return df.memory_usage(deep=True).sum() / 1024**2


def validate_and_clean_data(df: pd.DataFrame, required_columns: List[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate and clean financial data with memory optimization.

    Args:
        df: Input DataFrame
        required_columns: List of required columns

    Returns:
        Tuple of (is_valid: bool, errors: List[str])
    """
    errors = []

    if required_columns is None:
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]

    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return False, errors

    # Create copy to avoid warnings
    df_copy = df.copy()

    # Convert timestamp
    try:
        if not pd.api.types.is_datetime64_any_dtype(df_copy["timestamp"]):
            df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])
    except Exception as e:
        errors.append(f"Invalid timestamp format: {e}")
        return False, errors

    # Convert numeric columns and handle NaN
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df_copy.columns:
            try:
                pd.to_numeric(df_copy[col], errors="coerce")
            except Exception as e:
                errors.append(f"Invalid numeric data in column {col}: {e}")
                return False, errors

    # Check for negative prices
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if (df_copy[col] < 0).any():
            errors.append(f"Negative values found in {col} column")
            return False, errors

    # Check OHLC relationships
    if (df_copy["high"] < df_copy["low"]).any():
        errors.append("High price less than low price in some rows")
        return False, errors

    if (df_copy["close"] > df_copy["high"]).any() or (df_copy["close"] < df_copy["low"]).any():
        errors.append("Close price outside high-low range in some rows")
        return False, errors

    # Check for NaN values in critical columns
    critical_cols = ["timestamp", "close"]
    nan_counts = df_copy[critical_cols].isna().sum()
    if nan_counts.any():
        for col in critical_cols:
            if nan_counts[col] > 0:
                errors.append(f"NaN values found in critical column {col}: {nan_counts[col]} rows")
        return False, errors

    logger.info(f"Data validation successful: {len(df_copy)} rows")
    return True, errors


def resample_ohlcv(data: pd.DataFrame, timeframe: str, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Resample OHLCV data to a different timeframe.

    Args:
        data: OHLCV DataFrame
        timeframe: Target timeframe (e.g., '1H', '1D', '15min')
        timestamp_col: Name of timestamp column

    Returns:
        Resampled DataFrame
    """
    if data.empty:
        return data

    # Set timestamp as index for resampling
    df = data.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col)

    # Resample OHLCV data
    resampled = df.resample(timeframe).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})

    # Reset index and clean up
    resampled = resampled.dropna()
    resampled = resampled.reset_index()

    logger.info(f"Resampled {len(data)} records to {len(resampled)} records at {timeframe} interval")
    return resampled


def calculate_returns(prices: pd.Series, method: str = "simple", periods: int = 1) -> pd.Series:
    """
    Calculate price returns.

    Args:
        prices: Price series
        method: 'simple' or 'log'
        periods: Number of periods for return calculation

    Returns:
        Returns series
    """
    if method.lower() == "log":
        returns = np.log(prices / prices.shift(periods))
    else:  # simple returns
        returns = (prices / prices.shift(periods)) - 1

    return returns


def detect_outliers(data: pd.Series, method: str = "iqr", threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers in a data series.

    Args:
        data: Input data series
        method: Detection method ('iqr', 'zscore', 'modified_zscore')
        threshold: Threshold for outlier detection

    Returns:
        Boolean series indicating outliers
    """
    if method.lower() == "iqr":
        # Interquartile range method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (data < lower_bound) | (data > upper_bound)

    elif method.lower() == "zscore":
        # Z-score method
        mean_val = data.mean()
        std_val = data.std()
        if std_val == 0:
            return pd.Series(False, index=data.index)
        z_scores = np.abs((data - mean_val) / std_val)
        return z_scores > threshold

    elif method.lower() == "modified_zscore":
        # Modified Z-score method (more robust to outliers)
        median_val = data.median()
        mad = np.median(np.abs(data - median_val))
        if mad == 0:
            return pd.Series(False, index=data.index)
        modified_z = 0.6745 * (data - median_val) / mad
        return np.abs(modified_z) > threshold

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate common technical indicators.

    Args:
        data: OHLCV DataFrame

    Returns:
        DataFrame with additional technical indicators
    """
    df = data.copy()

    # Simple Moving Averages
    df["SMA_20"] = df["close"].rolling(window=20).mean()
    df["SMA_50"] = df["close"].rolling(window=50).mean()

    # Exponential Moving Averages
    df["EMA_12"] = df["close"].ewm(span=12).mean()
    df["EMA_26"] = df["close"].ewm(span=26).mean()

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_histogram"] = df["MACD"] - df["MACD_signal"]

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["BB_middle"] = df["close"].rolling(window=20).mean()
    df["BB_std"] = df["close"].rolling(window=20).std()
    df["BB_upper"] = df["BB_middle"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_middle"] - 2 * df["BB_std"]

    # Volume indicators
    df["Volume_SMA"] = df["volume"].rolling(window=20).mean()
    df["Volume_ratio"] = df["volume"] / df["Volume_SMA"]

    # Price momentum
    df["ROC_10"] = ((df["close"] - df["close"].shift(10)) / df["close"].shift(10)) * 100
    df["Momentum_10"] = df["close"] / df["close"].shift(10) - 1

    # Volatility
    df["Volatility_20"] = df["close"].pct_change().rolling(window=20).std() * np.sqrt(252)

    return df


def normalize_data(data: pd.DataFrame, method: str = "zscore", columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Normalize data using various methods.

    Args:
        data: DataFrame to normalize
        method: Normalization method ('zscore', 'minmax', 'robust')
        columns: Specific columns to normalize (default: all numeric)

    Returns:
        Normalized DataFrame
    """
    df = data.copy()

    if columns is None:
        # Normalize all numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
    else:
        numeric_columns = [col for col in columns if col in df.columns]

    for col in numeric_columns:
        if method == "zscore":
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val != 0:
                df[f"{col}_normalized"] = (df[col] - mean_val) / std_val
            else:
                df[f"{col}_normalized"] = 0

        elif method == "minmax":
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df[f"{col}_normalized"] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[f"{col}_normalized"] = 0.5

        elif method == "robust":
            median_val = df[col].median()
            mad = np.median(np.abs(df[col] - median_val))
            if mad != 0:
                df[f"{col}_normalized"] = (df[col] - median_val) / mad
            else:
                df[f"{col}_normalized"] = 0

    return df


def handle_missing_data(data: pd.DataFrame, method: str = "interpolate", columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Handle missing data in DataFrame.

    Args:
        data: DataFrame with missing data
        method: Method to handle missing data ('drop', 'fill', 'interpolate')
        columns: Specific columns to process

    Returns:
        DataFrame with handled missing data
    """
    df = data.copy()

    if columns is None:
        target_columns = df.columns
    else:
        target_columns = [col for col in columns if col in df.columns]

    if method == "drop":
        df = df.dropna(subset=target_columns)
    elif method == "fill":
        for col in target_columns:
            if df[col].dtype in ["float64", "int64"]:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "unknown")
    elif method == "interpolate":
        df[target_columns] = df[target_columns].interpolate(method="linear")

    logger.info(f"Handled missing data using {method} method")
    return df


def calculate_correlation_matrix(data: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate correlation matrix for specified columns.

    Args:
        data: Input DataFrame
        columns: Columns to include in correlation (default: numeric columns)

    Returns:
        Correlation matrix DataFrame
    """
    if columns is None:
        # Use all numeric columns
        corr_data = data.select_dtypes(include=[np.number])
    else:
        corr_data = data[columns]

    correlation_matrix = corr_data.corr()

    return correlation_matrix


def find_highly_correlated_pairs(correlation_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
    """
    Find highly correlated pairs in correlation matrix.

    Args:
        correlation_matrix: Correlation matrix DataFrame
        threshold: Correlation threshold

    Returns:
        List of tuples (col1, col2, correlation)
    """
    pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                pairs.append((col1, col2, corr_value))

    return pairs


def calculate_drawdowns(price_series: pd.Series) -> pd.Series:
    """
    Calculate drawdowns from a price series.

    Args:
        price_series: Series of prices

    Returns:
        Series of drawdowns
    """
    if price_series.empty:
        return pd.Series(dtype=float)

    # Calculate cumulative maximum
    cumulative_max = price_series.expanding().max()

    # Calculate drawdown
    drawdowns = (price_series - cumulative_max) / cumulative_max

    return drawdowns


def calculate_roll_max_drawdown(price_series: pd.Series, window: int = 252) -> pd.Series:
    """
    Calculate rolling maximum drawdown.

    Args:
        price_series: Series of prices
        window: Rolling window size

    Returns:
        Series of rolling maximum drawdowns
    """
    roll_max = price_series.rolling(window=window).max()
    roll_min = price_series.rolling(window=window).min()

    # Maximum drawdown in rolling window
    max_drawdowns = (roll_min - roll_max) / roll_max

    return max_drawdowns


def split_data_by_date(
    data: pd.DataFrame, split_date: str, date_column: str = "timestamp"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets by date.

    Args:
        data: Input DataFrame
        split_date: Date to split on (YYYY-MM-DD)
        date_column: Name of date column

    Returns:
        Tuple of (train_data, test_data)
    """
    split_timestamp = pd.Timestamp(split_date)

    train_data = data[data[date_column] < split_timestamp].copy()
    test_data = data[data[date_column] >= split_timestamp].copy()

    logger.info(f"Split data: {len(train_data)} train, {len(test_data)} test records")

    return train_data, test_data


def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calculate beta coefficient between asset and market returns.

    Args:
        asset_returns: Asset return series
        market_returns: Market return series

    Returns:
        Beta coefficient
    """
    # Align the series
    aligned_data = pd.concat([asset_returns, market_returns], axis=1).dropna()
    aligned_data.columns = ["asset", "market"]

    if len(aligned_data) < 2:
        return 1.0  # Default beta

    # Calculate covariance and variance
    covariance = aligned_data["asset"].cov(aligned_data["market"])
    market_variance = aligned_data["market"].var()

    if market_variance == 0:
        return 1.0

    beta = covariance / market_variance
    return beta


def calculate_alpha(asset_returns: pd.Series, market_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate alpha (excess return) of asset vs market.

    Args:
        asset_returns: Asset return series
        market_returns: Market return series
        risk_free_rate: Annual risk-free rate

    Returns:
        Alpha value
    """
    beta = calculate_beta(asset_returns, market_returns)

    # Annualize returns
    asset_return_annual = asset_returns.mean() * 252
    market_return_annual = market_returns.mean() * 252

    # Calculate alpha
    alpha = asset_return_annual - (risk_free_rate + beta * (market_return_annual - risk_free_rate))

    return alpha

