import logging
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


class MarketDataValidator(BaseModel):
    """
    Pydantic model for validating incoming market data.

    Enforces type safety and business rules.
    """

    symbol: str = Field(..., min_length=3, max_length=20)
    close: float = Field(..., gt=0)  # Must be positive
    timestamp: int = Field(..., gt=0)  # Unix timestamp

    @field_validator("symbol")
    @classmethod
    def validate_symbol_format(cls, v):
        """Ensure symbol is uppercase and contains no spaces."""
        if not v.isupper():
            raise ValueError("Symbol must be uppercase")
        if " " in v:
            raise ValueError("Symbol cannot contain spaces")
        return v

    @field_validator("close")
    @classmethod
    def validate_reasonable_price(cls, v):
        """Ensure price is within reasonable bounds (0.01 to 1M)."""
        if v < 0.01 or v > 1_000_000:
            raise ValueError("Price out of reasonable bounds")
        return v

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)


def validate_market_data(data: Dict[str, Any]) -> MarketDataValidator:
    """
    Validate market data dictionary and return validated model.

    Raises ValidationError if data is invalid.
    """
    try:
        return MarketDataValidator(**data)
    except ValidationError as e:
        logger.error(f"Market data validation failed: {e}")
        raise


class SignalValidator(BaseModel):
    """
    Validator for trading signals.
    """

    symbol: str = Field(..., min_length=3, max_length=20)
    side: str = Field(..., pattern=r"^(buy|sell)$")
    price: float = Field(..., gt=0)
    strength: float = Field(..., ge=0.0, le=1.0)

    model_config = ConfigDict(str_strip_whitespace=True)


def validate_signal(signal_data: Dict[str, Any]) -> SignalValidator:
    """
    Validate signal data.
    """
    try:
        return SignalValidator(**signal_data)
    except ValidationError as e:
        logger.error(f"Signal validation failed: {e}")
        raise
