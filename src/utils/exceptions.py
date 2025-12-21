#!/usr/bin/env python3
"""
Supreme System V5 - Custom Exceptions

Comprehensive exception hierarchy with advanced error handling,
decorators, and context management for production-grade reliability.
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import structlog

    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    structlog = None


class SupremeSystemError(Exception):
    """
    Base exception for Supreme System V5 with advanced context and logging.

    Provides structured error information, automatic logging, and recovery suggestions
    for production-grade error handling.
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None,
        log_level: str = "ERROR",
        details: Optional[Dict[str, Any]] = None,  # Backward compatibility
    ):
        super().__init__(message)

        self.message = message
        self.context = context or details or {}  # Support both context and details
        self.recovery_suggestions = recovery_suggestions or []
        self.log_level = log_level
        self.timestamp = datetime.now()
        self.error_code = error_code or self._generate_error_code()
        self.details = self.context  # Backward compatibility

        # Auto-log the error
        self._log_error()

    def _generate_error_code(self) -> str:
        """Generate a unique error code based on exception type."""
        class_name = self.__class__.__name__
        timestamp = self.timestamp.strftime("%H%M%S")
        return f"{class_name[:3].upper()}{timestamp}"

    def _log_error(self):
        """Log the error with context."""
        logger = logging.getLogger(self.__class__.__module__)

        log_message = f"[{self.error_code}] {self.message}"
        if self.context:
            log_message += f" | Context: {self.context}"

        if self.recovery_suggestions:
            log_message += f" | Suggestions: {self.recovery_suggestions}"

        logger.log(getattr(logging, self.log_level), log_message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "exception_type": self.__class__.__name__,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "recovery_suggestions": self.recovery_suggestions,
            "log_level": self.log_level,
        }

    def __str__(self) -> str:
        if self.context:
            return f"[{self.error_code}] {self.message} - Context: {self.context}"
        return f"[{self.error_code}] {self.message}"


class ConfigurationError(SupremeSystemError):
    """Raised when there are configuration-related errors."""

    pass


class DataError(SupremeSystemError):
    """Raised when there are data-related errors."""

    pass


class ValidationError(SupremeSystemError):
    """Raised when data validation fails."""

    pass


class TradingError(SupremeSystemError):
    """Raised when trading operations fail."""

    pass


class RiskError(SupremeSystemError):
    """Raised when risk management violations occur."""

    pass


class NetworkError(SupremeSystemError):
    """Raised when network/API operations fail."""

    pass


class StrategyError(SupremeSystemError):
    """Raised when strategy execution fails."""

    pass


class BacktestError(SupremeSystemError):
    """Raised when backtesting operations fail."""

    pass


class CircuitBreakerError(SupremeSystemError):
    """Raised when circuit breaker is triggered."""

    pass


class InsufficientFundsError(TradingError):
    """Raised when insufficient funds for trading."""

    pass


class InvalidOrderError(TradingError):
    """Raised when order parameters are invalid."""

    pass


class MarketDataError(DataError):
    """Raised when market data is unavailable or invalid."""

    pass


class ConnectionError(NetworkError):
    """Raised when connection to external services fails."""

    pass


class RateLimitError(NetworkError):
    """Raised when API rate limits are exceeded."""

    pass


class PositionSizeError(RiskError):
    """Raised when position size violates risk limits."""

    pass


class StopLossError(RiskError):
    """Raised when stop loss conditions are triggered."""

    pass


class MaxDrawdownError(RiskError):
    """Raised when maximum drawdown limits are exceeded."""

    pass


class StrategyTimeoutError(StrategyError):
    """Raised when strategy execution times out."""

    pass


class BacktestDataError(BacktestError):
    """Raised when backtest data is insufficient or invalid."""

    pass


class InsufficientDataError(DataError):
    """Raised when insufficient data for operations."""

    pass


class FileOperationError(SupremeSystemError):
    """Raised when file operations fail."""

    pass


class SerializationError(SupremeSystemError):
    """Raised when serialization/deserialization fails."""

    pass


class MonitoringError(SupremeSystemError):
    """Raised when monitoring operations fail."""

    pass


# Exception mapping for error handling
EXCEPTION_MAPPING = {
    "configuration": ConfigurationError,
    "data": DataError,
    "validation": ValidationError,
    "trading": TradingError,
    "risk": RiskError,
    "network": NetworkError,
    "strategy": StrategyError,
    "backtest": BacktestError,
    "circuit_breaker": CircuitBreakerError,
    "insufficient_funds": InsufficientFundsError,
    "invalid_order": InvalidOrderError,
    "market_data": MarketDataError,
    "connection": ConnectionError,
    "rate_limit": RateLimitError,
    "position_size": PositionSizeError,
    "stop_loss": StopLossError,
    "max_drawdown": MaxDrawdownError,
    "strategy_timeout": StrategyTimeoutError,
    "backtest_data": BacktestDataError,
    "insufficient_data": InsufficientDataError,
    "file_operation": FileOperationError,
    "serialization": SerializationError,
    "monitoring": MonitoringError,
}


def create_exception(
    error_type: str, message: str, details: Optional[Dict[str, Any]] = None
) -> SupremeSystemError:
    """
    Create an exception of the appropriate type.

    Args:
        error_type: Type of error (key from EXCEPTION_MAPPING)
        message: Error message
        details: Additional error details

    Returns:
        Appropriate exception instance

    Raises:
        ValueError: If error_type is not recognized
    """
    exception_class = EXCEPTION_MAPPING.get(error_type)
    if exception_class is None:
        raise ValueError(f"Unknown error type: {error_type}")

    return exception_class(message, details)


def handle_exception(exc: Exception, logger: Optional[Any] = None, re_raise: bool = True) -> None:
    """
    Handle an exception with proper logging and re-raising.

    Args:
        exc: Exception to handle
        logger: Logger instance for logging
        re_raise: Whether to re-raise the exception
    """
    if logger:
        if isinstance(exc, SupremeSystemError):
            logger.error(f"Supreme System Error: {exc}")
        else:
            logger.error(f"Unexpected error: {exc}", exc_info=True)

    if re_raise:
        raise exc


# Advanced Error Handling Decorators


def handle_errors(operation: str = None, log_level: str = "ERROR", reraise: bool = True):
    """
    Decorator for comprehensive error handling.

    Args:
        operation: Description of the operation
        log_level: Logging level for errors
        reraise: Whether to re-raise exceptions after handling

    Returns:
        Decorated function with error handling
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation or f"{func.__module__}.{func.__qualname__}"
            logger = logging.getLogger(func.__module__)

            try:
                logger.debug(f"Starting operation: {op_name}")
                result = func(*args, **kwargs)
                logger.debug(f"Operation completed: {op_name}")
                return result

            except SupremeSystemError as e:
                # Already logged by the exception itself
                if reraise:
                    raise
                return None

            except Exception as e:
                # Wrap unexpected exceptions
                error = SupremeSystemError(
                    f"Unexpected error in {op_name}: {str(e)}",
                    context={
                        "function": func.__qualname__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                    },
                    log_level=log_level,
                )

                if reraise:
                    raise error from e
                return None

        return wrapper

    return decorator


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (NetworkError, ConnectionError, RateLimitError),
):
    """
    Decorator for retrying operations on failure.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to retry on

    Returns:
        Decorated function with retry logic
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__qualname__}: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        import time

                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__qualname__}")

            raise last_exception

        return wrapper

    return decorator


class ErrorHandler:
    """
    Context manager for comprehensive error handling.

    Usage:
        with ErrorHandler("database operation", logger=my_logger):
            # risky code here
            pass
    """

    def __init__(
        self, operation: str, logger: Optional[logging.Logger] = None, reraise: bool = True
    ):
        """
        Initialize error handler.

        Args:
            operation: Description of the operation being performed
            logger: Logger instance (defaults to module logger)
            reraise: Whether to re-raise exceptions after handling
        """
        self.operation = operation
        self.logger = logger or logging.getLogger(__name__)
        self.reraise = reraise

    def __enter__(self):
        self.logger.debug(f"Starting operation: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Log the error with context
            if isinstance(exc_val, SupremeSystemError):
                self.logger.log(
                    getattr(logging, exc_val.log_level),
                    f"Operation failed: {self.operation} - {exc_val}",
                )
            else:
                self.logger.exception(
                    f"Operation failed: {self.operation} - Unexpected error: {exc_val}"
                )

            return not self.reraise  # Suppress exception if not reraising

        self.logger.debug(f"Operation completed successfully: {self.operation}")
        return False


# ===== ADVANCED ERROR HANDLING & RESILIENCE =====


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CircuitBreakerState(Enum):
    """Circuit breaker states for resilience patterns."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerOpenException(SupremeSystemError):
    """Exception raised when circuit breaker is open."""

    pass


class CircuitBreaker:
    """
    Circuit Breaker pattern implementation for fault tolerance.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: tuple = Exception,
        name: str = "default",
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._success_count = 0
        self._lock = threading.Lock()
        self.logger = logging.getLogger(f"circuit_breaker.{name}")

    def call(self, func: Callable, *args, **kwargs) -> Any:
        if self._state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._state = CircuitBreakerState.HALF_OPEN
            else:
                raise CircuitBreakerOpenException(f"Circuit breaker '{self.name}' is OPEN")

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.expected_exception as e:
            self._record_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        if self._last_failure_time is None:
            return False
        return time.time() - self._last_failure_time >= self.recovery_timeout

    def _record_success(self):
        with self._lock:
            self._success_count += 1
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.CLOSED
                self._failure_count = 0

    def _record_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitBreakerState.OPEN


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: tuple = Exception,
    name: Optional[str] = None,
):
    """Decorator to apply circuit breaker pattern."""

    def decorator(func: Callable) -> Callable:
        circuit_name = name or f"{func.__module__}.{func.__name__}"
        breaker = CircuitBreaker(
            failure_threshold, recovery_timeout, expected_exception, circuit_name
        )

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await breaker.call_async(func, *args, **kwargs)

            async def call_async(self, func, *args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except self.expected_exception as e:
                    self._record_failure()
                    raise
                else:
                    self._record_success()

            CircuitBreaker.call_async = call_async
            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return breaker.call(func, *args, **kwargs)

            return sync_wrapper

    return decorator


class ResilienceManager:
    """Centralized resilience management."""

    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger("resilience_manager")

    def register_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: tuple = Exception,
    ) -> CircuitBreaker:
        with self._lock:
            if name in self.circuit_breakers:
                return self.circuit_breakers[name]
            breaker = CircuitBreaker(failure_threshold, recovery_timeout, expected_exception, name)
            self.circuit_breakers[name] = breaker
            return breaker


resilience_manager = ResilienceManager()


def get_resilience_manager() -> ResilienceManager:
    return resilience_manager


def retry_with_circuit_breaker(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    circuit_breaker_name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
):
    """Enhanced retry decorator with circuit breaker."""

    def decorator(func: Callable) -> Callable:
        cb_name = circuit_breaker_name or f"{func.__module__}.{func.__name__}"
        circuit_breaker = resilience_manager.register_circuit_breaker(
            cb_name, failure_threshold, recovery_timeout, exceptions
        )

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return circuit_breaker.call(func, *args, **kwargs)
                except exceptions as e:
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff**attempt)
                        logging.warning(f"Retry attempt {attempt + 1} for {func.__name__}: {e}")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"All retry attempts exhausted for {func.__name__}: {e}")
                        raise
            raise Exception("Should not reach here")

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await circuit_breaker.call_async(func, *args, **kwargs)
                except exceptions as e:
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff**attempt)
                        logging.warning(
                            f"Async retry attempt {attempt + 1} for {func.__name__}: {e}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logging.error(
                            f"All async retry attempts exhausted for {func.__name__}: {e}"
                        )
                        raise
            raise Exception("Should not reach here")

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class RecoveryStrategy:
    """Base class for error recovery strategies."""

    def __init__(self, name: str):
        self.name = name

    def can_recover(self, error: Exception) -> bool:
        raise NotImplementedError

    def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        raise NotImplementedError


class FallbackRecoveryStrategy(RecoveryStrategy):
    def __init__(self, name: str, fallback_value: Any):
        super().__init__(name)
        self.fallback_value = fallback_value

    def can_recover(self, error: Exception) -> bool:
        return True

    def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        logging.info(f"Fallback recovery '{self.name}': using {self.fallback_value}")
        return self.fallback_value


class ErrorRecoveryManager:
    def __init__(self):
        self.strategies: List[RecoveryStrategy] = []
        self._default_strategies()
        self.logger = logging.getLogger("error_recovery")

    def _default_strategies(self):
        self.add_strategy(FallbackRecoveryStrategy("default_fallback", None))

    def add_strategy(self, strategy: RecoveryStrategy):
        self.strategies.append(strategy)

    def recover(self, error: Exception, context: Dict[str, Any] = None) -> Any:
        context = context or {}
        for strategy in self.strategies:
            if strategy.can_recover(error):
                try:
                    return strategy.recover(error, context)
                except Exception:
                    continue
        raise error


error_recovery_manager = ErrorRecoveryManager()


def get_error_recovery_manager() -> ErrorRecoveryManager:
    return error_recovery_manager


def resilient_operation(
    recovery_context: Optional[Dict[str, Any]] = None,
    enable_circuit_breaker: bool = True,
    **circuit_kwargs,
):
    """Decorator for resilient operations."""

    def decorator(func: Callable) -> Callable:
        if enable_circuit_breaker:
            func = circuit_breaker(**circuit_kwargs)(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            context = recovery_context or {}
            context.update({"function": func, "args": args, "kwargs": kwargs})
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return error_recovery_manager.recover(e, context)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            context = recovery_context or {}
            context.update({"function": func, "args": args, "kwargs": kwargs})
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                return error_recovery_manager.recover(e, context)

        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper

    return decorator


__all__ = [
    # Base exceptions
    "SupremeSystemError",
    # Specific exceptions
    "ConnectionError",
    "RateLimitError",
    "PositionSizeError",
    "StopLossError",
    "MaxDrawdownError",
    "StrategyTimeoutError",
    "BacktestDataError",
    "InsufficientDataError",
    "FileOperationError",
    "SerializationError",
    "MonitoringError",
    "EXCEPTION_MAPPING",
    "create_exception",
    "handle_exception",
    # Advanced error handling
    "handle_errors",
    "retry_on_failure",
    "ErrorHandler",
    # Resilience and fault tolerance
    "ErrorSeverity",
    "CircuitBreakerState",
    "CircuitBreakerOpenException",
    "CircuitBreaker",
    "circuit_breaker",
    "ResilienceManager",
    "get_resilience_manager",
    "retry_with_circuit_breaker",
    "RecoveryStrategy",
    "FallbackRecoveryStrategy",
    "RetryRecoveryStrategy",
    "ErrorRecoveryManager",
    "get_error_recovery_manager",
    "resilient_operation",
]
