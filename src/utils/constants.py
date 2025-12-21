#!/usr/bin/env python3
"""
PhiHorizon - Constants and Configuration

Shared constants, default configurations, and system-wide settings.
"""

from typing import Any, Dict, List

# Trading Constants
REQUIRED_OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

SUPPORTED_INTERVALS = [
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",  # Minutes
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",  # Hours
    "1d",
    "3d",  # Days
    "1w",
    "1M",  # Week, Month
]

SUPPORTED_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "ADAUSDT",
    "DOTUSDT",
    "LINKUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "SOLUSDT",
    "AVAXUSDT",
    "LTCUSDT",
]

# Risk Management Constants
DEFAULT_STOP_LOSS_PCT = 0.02  # 2%
DEFAULT_TAKE_PROFIT_PCT = 0.05  # 5%
DEFAULT_MAX_POSITION_SIZE_PCT = 0.10  # 10% of portfolio
DEFAULT_MAX_DAILY_LOSS_PCT = 0.05  # 5% daily loss limit
DEFAULT_MAX_PORTFOLIO_DRAWDOWN = 0.15  # 15% max drawdown

# Strategy Constants
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9
DEFAULT_RSI_PERIOD = 14
DEFAULT_BB_PERIOD = 20
DEFAULT_BB_STD = 2.0
DEFAULT_ATR_PERIOD = 14

# Data Pipeline Constants
DEFAULT_CACHE_TTL = 300  # 5 minutes
DEFAULT_MAX_DATA_AGE_HOURS = 24
DEFAULT_BATCH_SIZE = 1000
DEFAULT_MAX_CONCURRENT_REQUESTS = 5

# System Constants
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_CONFIG_UPDATE_INTERVAL = 60  # seconds
DEFAULT_HEALTH_CHECK_INTERVAL = 30  # seconds

# Performance Benchmarks
TARGET_RESPONSE_TIME_MS = 100
TARGET_MEMORY_USAGE_MB = 150
TARGET_TEST_COVERAGE_PCT = 80

# Default Configuration Template
DEFAULT_CONFIG = {
    # Trading Configuration
    "mode": "simulated",  # 'realtime' or 'simulated'
    "initial_capital": 10000.0,
    "symbols": SUPPORTED_SYMBOLS[:5],  # Default to top 5
    "base_currency": "USDT",
    # Strategy Configuration
    "strategies": ["momentum", "moving_average", "mean_reversion", "breakout"],
    "default_strategy_params": {
        "momentum": {
            "short_period": DEFAULT_MACD_FAST,
            "long_period": DEFAULT_MACD_SLOW,
            "signal_period": DEFAULT_MACD_SIGNAL,
            "trend_threshold": 0.02,
            "volume_confirmation": True,
        },
        "moving_average": {"short_period": 10, "long_period": 20, "method": "sma"},
        "mean_reversion": {"lookback_period": 20, "entry_threshold": 2.0, "exit_threshold": 0.5},
        "breakout": {"lookback_period": 20, "breakout_threshold": 0.02, "volume_multiplier": 1.5},
    },
    # Risk Management
    "risk_management": {
        "enabled": True,
        "stop_loss_pct": DEFAULT_STOP_LOSS_PCT,
        "take_profit_pct": DEFAULT_TAKE_PROFIT_PCT,
        "max_position_size_pct": DEFAULT_MAX_POSITION_SIZE_PCT,
        "max_daily_loss_pct": DEFAULT_MAX_DAILY_LOSS_PCT,
        "max_portfolio_drawdown": DEFAULT_MAX_PORTFOLIO_DRAWDOWN,
        "max_concurrent_positions": 5,
    },
    # Data Pipeline
    "data_pipeline": {
        "cache_enabled": True,
        "cache_ttl": DEFAULT_CACHE_TTL,
        "max_data_age_hours": DEFAULT_MAX_DATA_AGE_HOURS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "max_concurrent_requests": DEFAULT_MAX_CONCURRENT_REQUESTS,
        "compression": "snappy",
        "partition_by": "date",
    },
    # WebSocket Configuration
    "websocket": {
        "enabled": True,
        "max_connections": 10,
        "ping_interval": 20,
        "ping_timeout": 10,
        "reconnect_attempts": 5,
        "reconnect_delay": 5,
    },
    # Backtesting Configuration
    "backtesting": {
        "default_commission": 0.001,  # 0.1%
        "default_slippage": 0.0005,  # 0.05%
        "benchmark_symbol": "BTCUSDT",
        "walk_forward_window": 252,  # 1 year
        "walk_forward_step": 21,  # 1 month
    },
    # Live Trading Configuration
    "live_trading": {
        "enabled": False,
        "testnet": True,
        "max_order_value": 1000.0,  # Maximum single order value
        "min_order_value": 10.0,  # Minimum single order value
        "emergency_stop_enabled": True,
        "position_limits": {"max_positions": 10, "max_value_per_symbol": 5000.0},
    },
    # Monitoring Configuration
    "monitoring": {
        "enabled": True,
        "metrics_interval": 60,  # seconds
        "health_check_interval": DEFAULT_HEALTH_CHECK_INTERVAL,
        "alerts_enabled": True,
        "log_level": DEFAULT_LOG_LEVEL,
        "performance_tracking": True,
    },
    # System Configuration
    "system": {
        "max_memory_mb": TARGET_MEMORY_USAGE_MB,
        "max_cpu_percent": 80.0,
        "config_update_interval": DEFAULT_CONFIG_UPDATE_INTERVAL,
        "cleanup_interval_hours": 24,
        "backup_enabled": True,
    },
}

# Error Messages
ERROR_MESSAGES = {
    "missing_columns": "Missing required columns: {columns}",
    "invalid_data_type": "Invalid data type for column {column}: expected {expected}, got {actual}",
    "empty_dataset": "Dataset is empty",
    "invalid_symbol": "Invalid trading symbol: {symbol}",
    "insufficient_balance": "Insufficient balance for order: required {required}, available {available}",
    "risk_limit_exceeded": "Risk limit exceeded: {limit_type} = {value:.2%} (max allowed: {max_allowed:.2%})",
    "connection_failed": "Failed to connect to {service}: {error}",
    "validation_failed": "Data validation failed: {errors}",
    "storage_failed": "Data storage failed: {error}",
    "strategy_error": "Strategy {strategy} failed: {error}",
}

# Success Messages
SUCCESS_MESSAGES = {
    "data_loaded": "Successfully loaded {count} records for {symbol}",
    "order_placed": "Order placed successfully: {order_id} for {symbol} {side} {quantity} @ {price}",
    "position_closed": "Position closed: {symbol} P&L = {pnl:.2f} ({pnl_pct:.2%})",
    "strategy_signal": "Strategy {strategy} generated {signal} signal for {symbol}",
    "system_ready": "System initialized and ready for operation",
    "backup_completed": "Backup completed successfully: {files} files, {size} MB",
}

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
        "detailed": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "level": "INFO", "formatter": "standard", "stream": "ext://sys.stdout"},
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "logs/supreme_system.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file"]},
}

# Performance Metrics Thresholds
PERFORMANCE_THRESHOLDS = {
    "response_time_ms": TARGET_RESPONSE_TIME_MS,
    "memory_usage_mb": TARGET_MEMORY_USAGE_MB,
    "cpu_usage_percent": 80.0,
    "disk_usage_percent": 90.0,
    "network_latency_ms": 50,
    "error_rate_percent": 1.0,
    "uptime_percent": 99.9,
}

# Circuit Breaker Configuration
CIRCUIT_BREAKER_CONFIG = {
    "failure_threshold": 5,
    "timeout": 60,  # seconds
    "success_threshold": 3,
    "monitoring_interval": 10,  # seconds
    "auto_reset_enabled": True,
}

# Health Check Configuration
HEALTH_CHECK_CONFIG = {
    "database_connection": True,
    "api_endpoints": True,
    "websocket_connections": True,
    "memory_usage": True,
    "disk_space": True,
    "cpu_usage": True,
    "network_connectivity": True,
    "external_services": True,
}

# Export all constants
__all__ = [
    "REQUIRED_OHLCV_COLUMNS",
    "SUPPORTED_INTERVALS",
    "SUPPORTED_SYMBOLS",
    "DEFAULT_STOP_LOSS_PCT",
    "DEFAULT_TAKE_PROFIT_PCT",
    "DEFAULT_MAX_POSITION_SIZE_PCT",
    "DEFAULT_MAX_DAILY_LOSS_PCT",
    "DEFAULT_MAX_PORTFOLIO_DRAWDOWN",
    "DEFAULT_MACD_FAST",
    "DEFAULT_MACD_SLOW",
    "DEFAULT_MACD_SIGNAL",
    "DEFAULT_RSI_PERIOD",
    "DEFAULT_BB_PERIOD",
    "DEFAULT_BB_STD",
    "DEFAULT_ATR_PERIOD",
    "DEFAULT_CACHE_TTL",
    "DEFAULT_MAX_DATA_AGE_HOURS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_CONCURRENT_REQUESTS",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_CONFIG_UPDATE_INTERVAL",
    "DEFAULT_HEALTH_CHECK_INTERVAL",
    "TARGET_RESPONSE_TIME_MS",
    "TARGET_MEMORY_USAGE_MB",
    "TARGET_TEST_COVERAGE_PCT",
    "DEFAULT_CONFIG",
    "ERROR_MESSAGES",
    "SUCCESS_MESSAGES",
    "LOGGING_CONFIG",
    "PERFORMANCE_THRESHOLDS",
    "CIRCUIT_BREAKER_CONFIG",
    "HEALTH_CHECK_CONFIG",
]

