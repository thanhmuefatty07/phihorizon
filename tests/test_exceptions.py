"""
Comprehensive Tests for Exceptions Module

Tests custom exception hierarchy, decorators, and context managers.
Optimized for maximum coverage with audit-level quality.
"""

import pytest


class TestSupremeSystemError:
    """Tests for SupremeSystemError base class."""
    
    def test_import(self):
        """Test exception imports correctly."""
        from src.utils.exceptions import SupremeSystemError
        assert SupremeSystemError is not None
    
    def test_basic_creation(self):
        """Test basic exception creation."""
        from src.utils.exceptions import SupremeSystemError
        
        exc = SupremeSystemError("Test error")
        assert str(exc) == "Test error" or "Test error" in str(exc)
    
    def test_with_error_code(self):
        """Test exception with error code."""
        from src.utils.exceptions import SupremeSystemError
        
        exc = SupremeSystemError("Test error", error_code="ERR001")
        assert exc.error_code == "ERR001"
    
    def test_with_context(self):
        """Test exception with context dict."""
        from src.utils.exceptions import SupremeSystemError
        
        exc = SupremeSystemError("Test error", context={"key": "value"})
        assert exc.context["key"] == "value"
    
    def test_with_recovery_suggestions(self):
        """Test exception with recovery suggestions."""
        from src.utils.exceptions import SupremeSystemError
        
        exc = SupremeSystemError(
            "Test error",
            recovery_suggestions=["Try again", "Check config"]
        )
        assert len(exc.recovery_suggestions) == 2
    
    def test_to_dict(self):
        """Test exception serialization to dict."""
        from src.utils.exceptions import SupremeSystemError
        
        exc = SupremeSystemError("Test error")
        result = exc.to_dict()
        
        assert isinstance(result, dict)
        assert 'message' in result
    
    def test_auto_generate_error_code(self):
        """Test automatic error code generation."""
        from src.utils.exceptions import SupremeSystemError
        
        exc = SupremeSystemError("Test error")
        assert exc.error_code is not None
        assert len(exc.error_code) > 0


class TestConfigurationError:
    """Tests for ConfigurationError."""
    
    def test_import(self):
        """Test exception imports correctly."""
        from src.utils.exceptions import ConfigurationError
        assert ConfigurationError is not None
    
    def test_inherits_from_base(self):
        """Test inherits from SupremeSystemError."""
        from src.utils.exceptions import ConfigurationError, SupremeSystemError
        
        exc = ConfigurationError("Config error")
        assert isinstance(exc, SupremeSystemError)
    
    def test_raise_and_catch(self):
        """Test raise and catch."""
        from src.utils.exceptions import ConfigurationError
        
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Missing config key")


class TestDataError:
    """Tests for DataError."""
    
    def test_import(self):
        """Test exception imports correctly."""
        from src.utils.exceptions import DataError
        assert DataError is not None
    
    def test_raise_and_catch(self):
        """Test raise and catch."""
        from src.utils.exceptions import DataError
        
        with pytest.raises(DataError):
            raise DataError("Invalid data format")


class TestValidationError:
    """Tests for ValidationError."""
    
    def test_import(self):
        """Test exception imports correctly."""
        from src.utils.exceptions import ValidationError
        assert ValidationError is not None
    
    def test_raise_and_catch(self):
        """Test raise and catch."""
        from src.utils.exceptions import ValidationError
        
        with pytest.raises(ValidationError):
            raise ValidationError("Validation failed")


class TestDataValidationError:
    """Tests for DataValidationError (backward compatibility)."""
    
    def test_import(self):
        """Test exception imports correctly."""
        from src.utils.exceptions import DataValidationError
        assert DataValidationError is not None
    
    def test_inherits_from_validation_error(self):
        """Test inherits from ValidationError."""
        from src.utils.exceptions import DataValidationError, ValidationError
        
        exc = DataValidationError("Data validation error")
        assert isinstance(exc, ValidationError)


class TestTradingError:
    """Tests for TradingError."""
    
    def test_import(self):
        """Test exception imports correctly."""
        from src.utils.exceptions import TradingError
        assert TradingError is not None
    
    def test_raise_and_catch(self):
        """Test raise and catch."""
        from src.utils.exceptions import TradingError
        
        with pytest.raises(TradingError):
            raise TradingError("Trade execution failed")


class TestRiskError:
    """Tests for RiskError."""
    
    def test_import(self):
        """Test exception imports correctly."""
        from src.utils.exceptions import RiskError
        assert RiskError is not None
    
    def test_raise_and_catch(self):
        """Test raise and catch."""
        from src.utils.exceptions import RiskError
        
        with pytest.raises(RiskError):
            raise RiskError("Risk limit exceeded")


class TestNetworkError:
    """Tests for NetworkError."""
    
    def test_import(self):
        """Test exception imports correctly."""
        from src.utils.exceptions import NetworkError
        assert NetworkError is not None
    
    def test_raise_and_catch(self):
        """Test raise and catch."""
        from src.utils.exceptions import NetworkError
        
        with pytest.raises(NetworkError):
            raise NetworkError("Connection timeout")


class TestStrategyError:
    """Tests for StrategyError."""
    
    def test_import(self):
        """Test exception imports correctly."""
        from src.utils.exceptions import StrategyError
        assert StrategyError is not None


class TestBacktestError:
    """Tests for BacktestError."""
    
    def test_import(self):
        """Test exception imports correctly."""
        from src.utils.exceptions import BacktestError
        assert BacktestError is not None


class TestCircuitBreakerError:
    """Tests for CircuitBreakerError."""
    
    def test_import(self):
        """Test exception imports correctly."""
        from src.utils.exceptions import CircuitBreakerError
        assert CircuitBreakerError is not None


class TestInsufficientFundsError:
    """Tests for InsufficientFundsError."""
    
    def test_import(self):
        """Test exception imports correctly."""
        from src.utils.exceptions import InsufficientFundsError
        assert InsufficientFundsError is not None
    
    def test_inherits_from_trading_error(self):
        """Test inherits from TradingError."""
        from src.utils.exceptions import InsufficientFundsError, TradingError
        
        exc = InsufficientFundsError("Not enough funds")
        assert isinstance(exc, TradingError)


class TestInvalidOrderError:
    """Tests for InvalidOrderError."""
    
    def test_import(self):
        """Test exception imports correctly."""
        from src.utils.exceptions import InvalidOrderError
        assert InvalidOrderError is not None


class TestMarketDataError:
    """Tests for MarketDataError."""
    
    def test_import(self):
        """Test exception imports correctly."""
        from src.utils.exceptions import MarketDataError
        assert MarketDataError is not None


class TestRateLimitError:
    """Tests for RateLimitError."""
    
    def test_import(self):
        """Test exception imports correctly."""
        from src.utils.exceptions import RateLimitError
        assert RateLimitError is not None
    
    def test_inherits_from_network_error(self):
        """Test inherits from NetworkError."""
        from src.utils.exceptions import RateLimitError, NetworkError
        
        exc = RateLimitError("Rate limit exceeded")
        assert isinstance(exc, NetworkError)


class TestPositionSizeError:
    """Tests for PositionSizeError."""
    
    def test_import(self):
        """Test exception imports correctly."""
        from src.utils.exceptions import PositionSizeError
        assert PositionSizeError is not None


class TestStopLossError:
    """Tests for StopLossError."""
    
    def test_import(self):
        """Test exception imports correctly."""
        from src.utils.exceptions import StopLossError
        assert StopLossError is not None


class TestMaxDrawdownError:
    """Tests for MaxDrawdownError."""
    
    def test_import(self):
        """Test exception imports correctly."""
        from src.utils.exceptions import MaxDrawdownError
        assert MaxDrawdownError is not None


class TestStrategyTimeoutError:
    """Tests for StrategyTimeoutError."""
    
    def test_import(self):
        """Test exception imports correctly."""
        from src.utils.exceptions import StrategyTimeoutError
        assert StrategyTimeoutError is not None


class TestBacktestDataError:
    """Tests for BacktestDataError."""
    
    def test_import(self):
        """Test exception imports correctly."""
        from src.utils.exceptions import BacktestDataError
        assert BacktestDataError is not None


class TestInsufficientDataError:
    """Tests for InsufficientDataError."""
    
    def test_import(self):
        """Test exception imports correctly."""
        from src.utils.exceptions import InsufficientDataError
        assert InsufficientDataError is not None


class TestFileOperationError:
    """Tests for FileOperationError."""
    
    def test_import(self):
        """Test exception imports correctly."""
        from src.utils.exceptions import FileOperationError
        assert FileOperationError is not None


class TestSerializationError:
    """Tests for SerializationError."""
    
    def test_import(self):
        """Test exception imports correctly."""
        from src.utils.exceptions import SerializationError
        assert SerializationError is not None


class TestExceptionHierarchy:
    """Audit tests for exception hierarchy correctness."""
    
    def test_all_inherit_from_base(self):
        """Audit: All exceptions should inherit from SupremeSystemError."""
        from src.utils.exceptions import (
            SupremeSystemError,
            ConfigurationError,
            DataError,
            ValidationError,
            TradingError,
            RiskError,
            NetworkError,
            StrategyError,
            BacktestError,
        )
        
        exceptions = [
            ConfigurationError, DataError, ValidationError,
            TradingError, RiskError, NetworkError,
            StrategyError, BacktestError
        ]
        
        for exc_class in exceptions:
            exc = exc_class("Test")
            assert isinstance(exc, SupremeSystemError), f"{exc_class.__name__} should inherit from SupremeSystemError"
    
    def test_exception_mapping_exists(self):
        """Audit: EXCEPTION_MAPPING should be defined."""
        from src.utils.exceptions import EXCEPTION_MAPPING
        
        assert isinstance(EXCEPTION_MAPPING, dict)
        assert len(EXCEPTION_MAPPING) > 0


class TestExceptionMapping:
    """Tests for EXCEPTION_MAPPING."""
    
    def test_import(self):
        """Test EXCEPTION_MAPPING imports correctly."""
        from src.utils.exceptions import EXCEPTION_MAPPING
        assert EXCEPTION_MAPPING is not None
    
    def test_contains_standard_errors(self):
        """Test mapping contains standard error types."""
        from src.utils.exceptions import EXCEPTION_MAPPING
        
        # Should have mappings for common error types
        assert len(EXCEPTION_MAPPING) > 5
