#!/usr/bin/env python3
"""
PhiHorizon - CCXT Data Loader for OKX

Simplified real-time data loader using CCXT library.
Only OKX as data source (no Binance backup per user request).

Pipeline:
- OKX API: Real-time and validation data (2023+)
- Kaggle: Historical training data (2012-2021)
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OKXConfig:
    """Configuration for OKX data loader."""
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    max_candles_per_request: int = 1000  # OKX allows up to 1440
    rate_limit_delay: float = 0.25       # 10 req/2s = 0.2s between
    max_retries: int = 3
    retry_delay: float = 1.0


class CCXTLoader:
    """
    CCXT-based data loader for OKX.
    
    Usage:
        loader = CCXTLoader()
        df = loader.fetch_ohlcv('BTC/USDT', '1h', days=30)
    """
    
    def __init__(self, config: OKXConfig = None):
        self.config = config or OKXConfig()
        self._exchange = None
        self._ccxt_available = False
        self._init_exchange()
    
    def _init_exchange(self):
        """Initialize CCXT OKX exchange."""
        try:
            import ccxt
            self._exchange = ccxt.okx({
                'enableRateLimit': True,
                'rateLimit': 250,  # ms between requests
            })
            self._ccxt_available = True
            logger.info("CCXT OKX initialized successfully")
        except ImportError:
            logger.warning("CCXT not installed. Run: pip install ccxt")
            self._ccxt_available = False
        except Exception as e:
            logger.error(f"CCXT initialization error: {e}")
            self._ccxt_available = False
    
    @property
    def is_available(self) -> bool:
        """Check if CCXT is available."""
        return self._ccxt_available
    
    def fetch_ohlcv(
        self,
        symbol: str = None,
        timeframe: str = None,
        since: datetime = None,
        days: int = 30,
        limit: int = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from OKX.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1h', '4h', '1d')
            since: Start datetime (optional)
            days: Number of days to fetch if since not specified
            limit: Max candles to fetch
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if not self._ccxt_available:
            logger.error("CCXT not available")
            return None
        
        symbol = symbol or self.config.symbol
        timeframe = timeframe or self.config.timeframe
        
        # Calculate since timestamp
        if since is None:
            since = datetime.utcnow() - timedelta(days=days)
        
        since_ms = int(since.timestamp() * 1000)
        
        logger.info(f"Fetching {symbol} {timeframe} from {since}")
        
        all_candles = []
        current_since = since_ms
        max_candles = limit or (days * 24)  # hourly candles
        
        try:
            while len(all_candles) < max_candles:
                # Fetch batch
                candles = self._fetch_batch(
                    symbol, 
                    timeframe, 
                    current_since,
                    min(self.config.max_candles_per_request, max_candles - len(all_candles))
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # Update since for next batch
                current_since = candles[-1][0] + 1
                
                # Rate limiting
                time.sleep(self.config.rate_limit_delay)
                
                # Log progress
                if len(all_candles) % 1000 == 0:
                    logger.info(f"Fetched {len(all_candles)} candles...")
            
            if not all_candles:
                logger.warning("No data received from OKX")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_candles, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Ensure numeric types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} candles from OKX")
            return df
            
        except Exception as e:
            logger.error(f"OKX fetch error: {e}")
            return None
    
    def _fetch_batch(
        self,
        symbol: str,
        timeframe: str,
        since: int,
        limit: int,
    ) -> List:
        """Fetch a single batch of candles with retry."""
        for attempt in range(self.config.max_retries):
            try:
                candles = self._exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=since,
                    limit=limit
                )
                return candles
            except Exception as e:
                logger.warning(f"Batch fetch attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
        return []
    
    def fetch_recent(
        self,
        symbol: str = None,
        timeframe: str = None,
        limit: int = 100,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch most recent candles (shortcut method).
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            limit: Number of recent candles
            
        Returns:
            DataFrame with recent OHLCV data
        """
        if not self._ccxt_available:
            return None
        
        symbol = symbol or self.config.symbol
        timeframe = timeframe or self.config.timeframe
        
        try:
            candles = self._exchange.fetch_ohlcv(
                symbol,
                timeframe,
                limit=limit
            )
            
            if not candles:
                return None
            
            df = pd.DataFrame(candles, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"Recent fetch error: {e}")
            return None
    
    def get_exchange_info(self) -> Dict:
        """Get OKX exchange information."""
        if not self._ccxt_available:
            return {"error": "CCXT not available"}
        
        try:
            return {
                "id": self._exchange.id,
                "name": self._exchange.name,
                "has_ohlcv": self._exchange.has['fetchOHLCV'],
                "timeframes": list(self._exchange.timeframes.keys()),
                "rate_limit_ms": self._exchange.rateLimit,
            }
        except Exception as e:
            return {"error": str(e)}


class KaggleOKXHybridLoader:
    """
    Hybrid loader combining Kaggle historical + OKX real-time.
    
    Usage:
        loader = KaggleOKXHybridLoader()
        train_df = loader.load_training_data()  # Kaggle
        test_df = loader.load_validation_data()  # OKX
    """
    
    def __init__(
        self,
        kaggle_path: str = None,
        okx_config: OKXConfig = None,
    ):
        self.kaggle_path = kaggle_path
        self.okx_loader = CCXTLoader(okx_config)
    
    def load_training_data(
        self,
        kaggle_file: str = None,
    ) -> Optional[pd.DataFrame]:
        """
        Load training data from Kaggle dataset.
        
        For local use: Expects Kaggle data to be pre-downloaded.
        For Kaggle notebook: Auto-detects /kaggle/input paths.
        """
        import os
        import glob
        
        # Try Kaggle environment first
        if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            csv_files = glob.glob('/kaggle/input/**/*.csv', recursive=True)
            btc_files = [f for f in csv_files if 'btc' in f.lower() or 'bitcoin' in f.lower()]
            
            if btc_files:
                logger.info(f"Found Kaggle file: {btc_files[0]}")
                return self._load_csv(btc_files[0])
        
        # Try provided path
        if kaggle_file and os.path.exists(kaggle_file):
            return self._load_csv(kaggle_file)
        
        # Try local cache
        cache_paths = [
            '.cache/data/kaggle_btc.csv',
            'data/btc_historical.csv',
            'kaggle_data.csv',
        ]
        
        for path in cache_paths:
            if os.path.exists(path):
                return self._load_csv(path)
        
        logger.warning("No Kaggle data found. Use OKX for all data.")
        return None
    
    def load_validation_data(
        self,
        days: int = 60,
    ) -> Optional[pd.DataFrame]:
        """
        Load validation data from OKX (recent data).
        
        Args:
            days: Number of days of recent data
            
        Returns:
            DataFrame with recent OHLCV data
        """
        return self.okx_loader.fetch_ohlcv(days=days)
    
    def load_hybrid(
        self,
        kaggle_file: str = None,
        validation_days: int = 60,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load both training (Kaggle) and validation (OKX) data.
        
        Returns:
            (training_df, validation_df)
        """
        train_df = self.load_training_data(kaggle_file)
        val_df = self.load_validation_data(days=validation_days)
        
        return train_df, val_df
    
    def _load_csv(self, path: str) -> pd.DataFrame:
        """Load and standardize CSV file."""
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Rename common variants
        rename_map = {
            'datetime': 'timestamp',
            'date': 'timestamp',
            'time': 'timestamp',
            'price': 'close',
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        # Ensure required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                if col == 'volume':
                    df[col] = 0
                else:
                    df[col] = df.get('close', 0)
        
        # Convert types
        for col in required:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        return df.dropna(subset=['close'])


# Convenience functions
def load_okx_data(symbol: str = "BTC/USDT", days: int = 30) -> Optional[pd.DataFrame]:
    """Quick function to load OKX data."""
    loader = CCXTLoader()
    return loader.fetch_ohlcv(symbol=symbol, days=days)


def load_hybrid_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Quick function to load hybrid Kaggle + OKX data."""
    loader = KaggleOKXHybridLoader()
    return loader.load_hybrid()


__all__ = [
    'OKXConfig',
    'CCXTLoader',
    'KaggleOKXHybridLoader',
    'load_okx_data',
    'load_hybrid_data',
]
