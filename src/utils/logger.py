"""
Comprehensive logging configuration for PhiHorizon.

Provides structured logging with file rotation, console output,
and configurable log levels for production-grade error handling.
"""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_level: Optional[str] = None,
    file_level: Optional[str] = None,
) -> logging.Logger:
    """
    Setup comprehensive logging configuration.

    Args:
        level: Default logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path. If None, creates timestamped file in logs/
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        console_level: Console logging level (defaults to main level)
        file_level: File logging level (defaults to main level)

    Returns:
        Root logger configured with comprehensive settings
    """
    # Convert string levels to logging constants
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    log_level = level_map.get(level.upper(), logging.INFO)
    console_log_level = level_map.get(console_level.upper() if console_level else level.upper(), log_level)
    file_log_level = level_map.get(file_level.upper() if file_level else level.upper(), log_level)

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create formatter with structured information
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Setup handlers list
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # File handler with rotation
    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(exist_ok=True)
    else:
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = log_dir / f"supreme_system_{timestamp}.log"

    file_handler = logging.handlers.RotatingFileHandler(
        file_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(file_log_level)
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add new handlers
    for handler in handlers:
        root_logger.addHandler(handler)

    # Log initial setup
    root_logger.info(f"Logging initialized - Level: {level}, File: {file_path}")
    root_logger.info(f"Log rotation: {max_bytes} bytes, {backup_count} backups")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_function_call(logger: logging.Logger, func_name: str, args: tuple = None, kwargs: dict = None):
    """
    Log function entry with parameters.

    Args:
        logger: Logger instance
        func_name: Function name
        args: Positional arguments
        kwargs: Keyword arguments
    """
    if logger.isEnabledFor(logging.DEBUG):
        args_str = f"args={args}" if args else ""
        kwargs_str = f"kwargs={kwargs}" if kwargs else ""
        params = ", ".join(filter(None, [args_str, kwargs_str]))
        logger.debug(f"Entering {func_name}({params})")


def log_function_result(logger: logging.Logger, func_name: str, result: Any, execution_time: float = None):
    """
    Log function exit with result.

    Args:
        logger: Logger instance
        func_name: Function name
        result: Function result
        execution_time: Optional execution time in seconds
    """
    if logger.isEnabledFor(logging.DEBUG):
        time_str = f" in {execution_time:.3f}s" if execution_time else ""
        logger.debug(f"Exiting {func_name}, result={result}{time_str}")


def log_error(logger: logging.Logger, error: Exception, context: str = "", include_traceback: bool = True):
    """
    Log exception with context.

    Args:
        logger: Logger instance
        error: Exception object
        context: Additional context information
        include_traceback: Whether to include full traceback
    """
    error_msg = f"{type(error).__name__}: {str(error)}"
    if context:
        error_msg = f"{context} - {error_msg}"

    if include_traceback:
        logger.exception(error_msg)
    else:
        logger.error(error_msg)


class ErrorHandler:
    """Context manager for comprehensive error handling."""

    def __init__(self, logger: logging.Logger, operation: str, raise_exception: bool = True):
        """
        Initialize error handler.

        Args:
            logger: Logger instance
            operation: Description of the operation being performed
            raise_exception: Whether to re-raise the exception after logging
        """
        self.logger = logger
        self.operation = operation
        self.raise_exception = raise_exception

    def __enter__(self):
        self.logger.debug(f"Starting operation: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            log_error(self.logger, exc_val, f"Operation failed: {self.operation}")
            return not self.raise_exception  # Suppress exception if not raising

        self.logger.debug(f"Operation completed successfully: {self.operation}")
        return False


def setup_trading_logging() -> logging.Logger:
    """
    Setup logging specifically configured for trading systems.

    Returns:
        Configured logger for trading operations
    """
    return setup_logging(
        level="INFO", console_level="WARNING", file_level="DEBUG"  # Less verbose on console  # Detailed logging to file
    )


def create_performance_logger(name: str = "performance") -> logging.Logger:
    """
    Create a specialized logger for performance monitoring.

    Args:
        name: Logger name

    Returns:
        Performance logger instance
    """
    logger = logging.getLogger(name)

    # Only add handler if not already added
    if not logger.handlers:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Performance-specific formatter
        perf_formatter = logging.Formatter(
            "%(asctime)s - PERFORMANCE - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File handler for performance logs
        perf_file = log_dir / "performance.log"
        perf_handler = logging.handlers.RotatingFileHandler(perf_file, maxBytes=50 * 1024 * 1024, backupCount=10)  # 50MB
        perf_handler.setFormatter(perf_formatter)
        perf_handler.setLevel(logging.INFO)

        logger.addHandler(perf_handler)
        logger.setLevel(logging.INFO)

        # Prevent propagation to root logger
        logger.propagate = False

    return logger

