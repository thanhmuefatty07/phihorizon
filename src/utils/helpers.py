#!/usr/bin/env python3
"""
PhiHorizon - Helper Utilities

Common helper functions used across the system.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd


def setup_logging(level: str = "INFO", log_file: Optional[str] = None, format_string: Optional[str] = None) -> logging.Logger:
    """
    Setup standardized logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        format_string: Optional custom format string

    Returns:
        Root logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(level=numeric_level, format=format_string, handlers=[])

    # Get root logger
    logger = logging.getLogger()

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate configuration dictionary has required keys.

    Args:
        config: Configuration dictionary
        required_keys: List of required keys

    Returns:
        True if valid, False otherwise
    """
    missing_keys = []
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
        elif config[key] is None:
            missing_keys.append(key)

    if missing_keys:
        logger = logging.getLogger(__name__)
        logger.error(f"Missing required configuration keys: {missing_keys}")
        return False

    return True


def safe_divide(numerator: Union[float, int], denominator: Union[float, int], default: float = 0.0) -> float:
    """
    Safe division that handles division by zero.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return on division by zero

    Returns:
        Division result or default value
    """
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        return float(numerator) / float(denominator)
    except (ZeroDivisionError, TypeError, ValueError):
        return default


def calculate_percentage_change(old_value: Union[float, int], new_value: Union[float, int]) -> float:
    """
    Calculate percentage change between two values.

    Args:
        old_value: Original value
        new_value: New value

    Returns:
        Percentage change (e.g., 0.05 for 5% increase)
    """
    return safe_divide(new_value - old_value, old_value, 0.0)


def calculate_moving_average(data: pd.Series, window: int, method: str = "sma") -> pd.Series:
    """
    Calculate moving average with different methods.

    Args:
        data: Input data series
        window: Moving average window
        method: 'sma' (simple), 'ema' (exponential), 'wma' (weighted)

    Returns:
        Moving average series
    """
    if method.lower() == "sma":
        return data.rolling(window=window).mean()
    elif method.lower() == "ema":
        return data.ewm(span=window).mean()
    elif method.lower() == "wma":
        # Vectorized WMA calculation using convolution
        weights = np.arange(1, window + 1, dtype=np.float64)
        weights = weights / weights.sum()  # Normalize weights

        # Use numpy.convolve for efficient WMA calculation
        def wma_conv(series):
            return np.convolve(series.values, weights[::-1], mode="valid")

        result = data.rolling(window=window).apply(lambda x: np.sum(x * weights), raw=False)
        return result
    else:
        raise ValueError(f"Unknown moving average method: {method}")


def calculate_volatility(data: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """
    Calculate rolling volatility (standard deviation).

    Args:
        data: Input data series (typically returns)
        window: Rolling window size
        annualize: Whether to annualize volatility

    Returns:
        Volatility series
    """
    vol = data.rolling(window=window).std()

    if annualize:
        # Assuming daily data, annualize by sqrt(252)
        vol = vol * np.sqrt(252)

    return vol


def detect_data_gaps(data: pd.DataFrame, timestamp_col: str = "timestamp", max_gap_minutes: int = 60) -> pd.DataFrame:
    """
    Detect time gaps in data that exceed maximum allowed gap.

    Args:
        data: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        max_gap_minutes: Maximum allowed gap in minutes

    Returns:
        DataFrame with gap information
    """
    if timestamp_col not in data.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found")

    # Ensure timestamps are sorted
    df = data.copy().sort_values(timestamp_col)

    # Calculate time differences
    time_diffs = df[timestamp_col].diff()

    # Convert to minutes
    gap_minutes = time_diffs.dt.total_seconds() / 60

    # Find gaps exceeding threshold
    significant_gaps = gap_minutes > max_gap_minutes

    gap_info = pd.DataFrame({"timestamp": df[timestamp_col], "gap_minutes": gap_minutes, "significant_gap": significant_gaps})

    return gap_info[significant_gaps]


def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return ".1f"
        size_bytes /= 1024.0
    return ".1f"


def validate_ohlcv_columns(data: pd.DataFrame) -> bool:
    """
    Validate that DataFrame has required OHLCV columns.

    Args:
        data: DataFrame to validate

    Returns:
        True if all required columns exist
    """
    required_columns = ["timestamp", "open", "high", "low", "close", "volume"]

    for col in required_columns:
        if col not in data.columns:
            logger = logging.getLogger(__name__)
            logger.error(f"Missing required column: {col}")
            return False

    return True


def normalize_symbol(symbol: str) -> str:
    """
    Normalize trading symbol to consistent format.

    Args:
        symbol: Raw symbol string

    Returns:
        Normalized symbol
    """
    # Convert to uppercase and remove common separators
    normalized = symbol.upper().replace("/", "").replace("-", "")

    # Ensure proper format (BASEQUOTE)
    if len(normalized) > 6:  # Likely has extra characters
        # Try to extract base and quote
        # This is a simple heuristic - could be improved
        if "USDT" in normalized:
            base = normalized.replace("USDT", "")
            normalized = f"{base}USDT"
        elif "BTC" in normalized:
            base = normalized.replace("BTC", "")
            normalized = f"{base}BTC"

    return normalized


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, annualize: bool = True) -> float:
    """
    Calculate Sharpe ratio for a series of returns.

    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate (default 2%)
        annualize: Whether to annualize the ratio

    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    # Calculate excess returns
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate

    # Calculate annualized Sharpe ratio
    if annualize:
        mean_excess_return = excess_returns.mean() * 252
        volatility = excess_returns.std() * np.sqrt(252)
    else:
        mean_excess_return = excess_returns.mean()
        volatility = excess_returns.std()

    if volatility == 0:
        return 0.0

    return mean_excess_return / volatility


def calculate_max_drawdown(price_series: pd.Series) -> float:
    """
    Calculate maximum drawdown from a price series.

    Args:
        price_series: Series of prices

    Returns:
        Maximum drawdown as decimal (e.g., 0.15 for 15%)
    """
    if len(price_series) < 2:
        return 0.0

    # Calculate cumulative maximum
    cumulative_max = price_series.expanding().max()

    # Calculate drawdown
    drawdown = (price_series - cumulative_max) / cumulative_max

    # Return maximum drawdown (most negative value)
    return abs(drawdown.min())


def round_to_significant_digits(value: float, digits: int = 4) -> float:
    """
    Round a number to a specified number of significant digits.

    Args:
        value: Number to round
        digits: Number of significant digits

    Returns:
        Rounded number
    """
    if value == 0:
        return 0.0

    import math

    return round(value, digits - int(math.floor(math.log10(abs(value)))) - 1)

