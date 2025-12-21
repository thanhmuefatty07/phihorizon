"""
Utility Functions Tests

Tests for utility modules in src/utils.
"""

import pytest
import numpy as np
import pandas as pd


class TestExceptions:
    """Tests for custom exceptions."""
    
    def test_import_exceptions(self):
        """Test that exceptions can be imported."""
        from src.utils.exceptions import (
            SupremeSystemError,
            ConfigurationError,
            DataValidationError,
        )
        assert SupremeSystemError is not None


class TestConstants:
    """Tests for constants module."""
    
    def test_import_constants(self):
        """Test that constants can be imported."""
        from src.utils import constants
        assert constants is not None


class TestHelpers:
    """Tests for helper functions."""
    
    def test_import_helpers(self):
        """Test that helpers can be imported."""
        from src.utils import helpers
        assert helpers is not None


class TestDataUtils:
    """Tests for data utility functions."""
    
    def test_import_data_utils(self):
        """Test that data_utils can be imported."""
        from src.utils import data_utils
        assert data_utils is not None


class TestVectorizedOps:
    """Tests for vectorized operations."""
    
    def test_import_vectorized_ops(self):
        """Test that vectorized_ops can be imported."""
        from src.utils import vectorized_ops
        assert vectorized_ops is not None


class TestFastMath:
    """Tests for fast math operations."""
    
    def test_import_fast_math(self):
        """Test that fast_math can be imported."""
        from src.utils import fast_math
        assert fast_math is not None
