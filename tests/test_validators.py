"""
Comprehensive Tests for Validators Module

Tests for MarketDataValidator, SignalValidator and validation functions.
Optimized for maximum coverage with edge cases.
"""

import pytest
from pydantic import ValidationError


class TestMarketDataValidator:
    """Tests for MarketDataValidator class."""
    
    def test_import(self):
        """Test module imports correctly."""
        from src.utils.validators import MarketDataValidator
        assert MarketDataValidator is not None
    
    def test_valid_market_data(self):
        """Test valid market data passes validation."""
        from src.utils.validators import MarketDataValidator
        
        data = MarketDataValidator(
            symbol="BTCUSDT",
            close=45000.0,
            timestamp=1703260800
        )
        assert data.symbol == "BTCUSDT"
        assert data.close == 45000.0
    
    def test_symbol_must_be_uppercase(self):
        """Test symbol validation requires uppercase."""
        from src.utils.validators import MarketDataValidator
        
        with pytest.raises(ValidationError) as exc_info:
            MarketDataValidator(
                symbol="btcusdt",  # lowercase
                close=45000.0,
                timestamp=1703260800
            )
        assert "uppercase" in str(exc_info.value).lower()
    
    def test_symbol_no_spaces(self):
        """Test symbol cannot contain spaces."""
        from src.utils.validators import MarketDataValidator
        
        with pytest.raises(ValidationError):
            MarketDataValidator(
                symbol="BTC USDT",  # has space
                close=45000.0,
                timestamp=1703260800
            )
    
    def test_close_must_be_positive(self):
        """Test close price must be positive."""
        from src.utils.validators import MarketDataValidator
        
        with pytest.raises(ValidationError):
            MarketDataValidator(
                symbol="BTCUSDT",
                close=-100.0,  # negative
                timestamp=1703260800
            )
    
    def test_close_reasonable_bounds_low(self):
        """Test close price must be >= 0.01."""
        from src.utils.validators import MarketDataValidator
        
        with pytest.raises(ValidationError):
            MarketDataValidator(
                symbol="BTCUSDT",
                close=0.001,  # too low
                timestamp=1703260800
            )
    
    def test_close_reasonable_bounds_high(self):
        """Test close price must be <= 1,000,000."""
        from src.utils.validators import MarketDataValidator
        
        with pytest.raises(ValidationError):
            MarketDataValidator(
                symbol="BTCUSDT",
                close=2_000_000.0,  # too high
                timestamp=1703260800
            )
    
    def test_timestamp_must_be_positive(self):
        """Test timestamp must be positive."""
        from src.utils.validators import MarketDataValidator
        
        with pytest.raises(ValidationError):
            MarketDataValidator(
                symbol="BTCUSDT",
                close=45000.0,
                timestamp=-1
            )


class TestSignalValidator:
    """Tests for SignalValidator class."""
    
    def test_import(self):
        """Test module imports correctly."""
        from src.utils.validators import SignalValidator
        assert SignalValidator is not None
    
    def test_valid_buy_signal(self):
        """Test valid buy signal passes validation."""
        from src.utils.validators import SignalValidator
        
        signal = SignalValidator(
            symbol="BTCUSDT",
            side="buy",
            price=45000.0,
            strength=0.85
        )
        assert signal.side == "buy"
        assert signal.strength == 0.85
    
    def test_valid_sell_signal(self):
        """Test valid sell signal passes validation."""
        from src.utils.validators import SignalValidator
        
        signal = SignalValidator(
            symbol="ETHUSDT",
            side="sell",
            price=2500.0,
            strength=0.65
        )
        assert signal.side == "sell"
    
    def test_invalid_side(self):
        """Test only buy/sell allowed for side."""
        from src.utils.validators import SignalValidator
        
        with pytest.raises(ValidationError):
            SignalValidator(
                symbol="BTCUSDT",
                side="hold",  # invalid
                price=45000.0,
                strength=0.5
            )
    
    def test_strength_range_low(self):
        """Test strength must be >= 0."""
        from src.utils.validators import SignalValidator
        
        with pytest.raises(ValidationError):
            SignalValidator(
                symbol="BTCUSDT",
                side="buy",
                price=45000.0,
                strength=-0.1  # negative
            )
    
    def test_strength_range_high(self):
        """Test strength must be <= 1."""
        from src.utils.validators import SignalValidator
        
        with pytest.raises(ValidationError):
            SignalValidator(
                symbol="BTCUSDT",
                side="buy",
                price=45000.0,
                strength=1.5  # too high
            )
    
    def test_strength_boundary_zero(self):
        """Test strength can be exactly 0."""
        from src.utils.validators import SignalValidator
        
        signal = SignalValidator(
            symbol="BTCUSDT",
            side="buy",
            price=45000.0,
            strength=0.0
        )
        assert signal.strength == 0.0
    
    def test_strength_boundary_one(self):
        """Test strength can be exactly 1."""
        from src.utils.validators import SignalValidator
        
        signal = SignalValidator(
            symbol="BTCUSDT",
            side="sell",
            price=45000.0,
            strength=1.0
        )
        assert signal.strength == 1.0


class TestValidationFunctions:
    """Tests for validation convenience functions."""
    
    def test_validate_market_data_valid(self):
        """Test validate_market_data with valid data."""
        from src.utils.validators import validate_market_data
        
        data = {
            "symbol": "BTCUSDT",
            "close": 45000.0,
            "timestamp": 1703260800
        }
        result = validate_market_data(data)
        assert result.symbol == "BTCUSDT"
    
    def test_validate_market_data_invalid(self):
        """Test validate_market_data with invalid data raises."""
        from src.utils.validators import validate_market_data
        
        data = {
            "symbol": "invalid",  # lowercase
            "close": 45000.0,
            "timestamp": 1703260800
        }
        with pytest.raises(ValidationError):
            validate_market_data(data)
    
    def test_validate_signal_valid(self):
        """Test validate_signal with valid data."""
        from src.utils.validators import validate_signal
        
        data = {
            "symbol": "BTCUSDT",
            "side": "buy",
            "price": 45000.0,
            "strength": 0.8
        }
        result = validate_signal(data)
        assert result.side == "buy"
    
    def test_validate_signal_invalid(self):
        """Test validate_signal with invalid data raises."""
        from src.utils.validators import validate_signal
        
        data = {
            "symbol": "BTCUSDT",
            "side": "invalid",
            "price": 45000.0,
            "strength": 0.8
        }
        with pytest.raises(ValidationError):
            validate_signal(data)
