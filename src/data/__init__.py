"""
PhiHorizon - Data Module

Hybrid data pipeline:
- Kaggle datasets for training/research (reproducible)
- OKX API for validation/production (real-time via CCXT)
"""

from .hybrid_loader import (
    DataConfig,
    DataPurpose,
    DataQualityReport,
    DataSource,
    HybridDataLoader,
    load_data_for_research,
    load_data_for_validation,
)

from .ccxt_loader import (
    CCXTLoader,
    KaggleOKXHybridLoader,
    OKXConfig,
    load_hybrid_data,
    load_okx_data,
)

__all__ = [
    # Original hybrid loader
    'DataSource',
    'DataPurpose',
    'DataConfig',
    'DataQualityReport',
    'HybridDataLoader',
    'load_data_for_research',
    'load_data_for_validation',
    # New CCXT/OKX loader
    'OKXConfig',
    'CCXTLoader',
    'KaggleOKXHybridLoader',
    'load_okx_data',
    'load_hybrid_data',
]
