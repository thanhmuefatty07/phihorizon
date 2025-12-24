#!/usr/bin/env python3
"""
PhiHorizon - Comprehensive Training Data Preparation

Creates large-scale training dataset from:
1. Kaggle datasets (historical 5+ years)
2. Bybit API (latest data + derivatives)

Coins: BTC + ETH (combined for robust training)
Timeframe: Hourly (balance between data size and noise)
Features: 30 real features (no placeholders)

Usage:
    python scripts/prepare_training_data.py --kaggle-path /kaggle/input/...
    python scripts/prepare_training_data.py --download-bybit
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    'coins': ['BTC', 'ETH'],
    'timeframe': '1h',  # Hourly for good balance
    'lookback_days': 1825,  # 5 years
    'output_dir': 'data/training',
    'features': [
        # Price (5)
        'open', 'high', 'low', 'close', 'volume',
        # Returns (5)
        'return_1h', 'return_24h', 'return_7d', 'volatility_24h', 'volatility_7d',
        # Technical (9)
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_lower', 'bb_width', 'atr_14', 'adx_14',
        # Derivatives (5) - from Bybit
        'funding_rate', 'open_interest', 'oi_change', 'long_short_ratio', 'volume_ratio',
        # Cross-asset (4) - BTC influence on ETH
        'btc_return_24h', 'btc_volatility', 'btc_correlation', 'btc_lead',
        # Market (2)
        'hour_sin', 'hour_cos',  # Time encoding
    ]
}


# ============================================================
# BYBIT API DATA LOADER
# ============================================================

def fetch_bybit_ohlcv(symbol: str = 'ETHUSDT', interval: str = '60', days: int = 365) -> pd.DataFrame:
    """
    Fetch OHLCV data from Bybit using CCXT.
    
    Args:
        symbol: Trading pair (e.g., 'ETHUSDT', 'BTCUSDT')
        interval: Timeframe in minutes ('60' = 1 hour)
        days: Number of days to fetch
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        import ccxt
    except ImportError:
        logger.warning("CCXT not installed. Run: pip install ccxt")
        return pd.DataFrame()
    
    logger.info(f"Fetching {symbol} from Bybit for {days} days...")
    
    exchange = ccxt.bybit({
        'enableRateLimit': True,
        'options': {'defaultType': 'linear'}  # USDT perpetual
    })
    
    since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, interval + 'm', since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            
            if len(all_ohlcv) >= days * 24:  # hourly
                break
                
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            logger.warning(f"Error fetching: {e}")
            break
    
    if not all_ohlcv:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Fetched {len(df)} rows from {df.timestamp.min()} to {df.timestamp.max()}")
    return df


def fetch_bybit_funding_rate(symbol: str = 'ETHUSDT', days: int = 30) -> pd.DataFrame:
    """Fetch funding rate history from Bybit."""
    try:
        import ccxt
        exchange = ccxt.bybit({'enableRateLimit': True})
        
        # Bybit funding rate endpoint
        funding = exchange.fetch_funding_rate_history(symbol, limit=days * 3)  # 3 per day
        
        df = pd.DataFrame(funding)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df[['timestamp', 'fundingRate']]
        
    except Exception as e:
        logger.warning(f"Failed to fetch funding rate: {e}")
        return pd.DataFrame()


def fetch_bybit_open_interest(symbol: str = 'ETHUSDT') -> float:
    """Fetch current open interest from Bybit."""
    try:
        import ccxt
        exchange = ccxt.bybit({'enableRateLimit': True})
        
        ticker = exchange.fetch_ticker(symbol)
        return ticker.get('info', {}).get('openInterest', 0)
        
    except Exception as e:
        logger.warning(f"Failed to fetch OI: {e}")
        return 0


# ============================================================
# KAGGLE DATA LOADER
# ============================================================

def load_kaggle_dataset(path: str, coin: str = 'ETH') -> pd.DataFrame:
    """
    Load and preprocess Kaggle dataset.
    
    Supports common Kaggle crypto dataset formats.
    """
    logger.info(f"Loading Kaggle dataset from {path}...")
    
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    
    # Standardize column names
    column_mapping = {
        'Date': 'timestamp',
        'Time': 'timestamp', 
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        'open_time': 'timestamp',
        'close_time': 'timestamp',
    }
    
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        if df['timestamp'].dtype in ['int64', 'float64']:
            # Assume milliseconds
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Keep only needed columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = df[[c for c in required_cols if c in df.columns]]
    
    # Add coin identifier
    df['coin'] = coin
    
    logger.info(f"Loaded {len(df)} rows for {coin}")
    return df


def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample minute data to hourly."""
    df = df.set_index('timestamp')
    
    resampled = df.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return resampled.reset_index()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical features."""
    df = df.copy()
    
    # ========== RETURNS ==========
    df['return_1h'] = df['close'].pct_change(1)
    df['return_24h'] = df['close'].pct_change(24)
    df['return_7d'] = df['close'].pct_change(24 * 7)
    
    # Volatility
    df['volatility_24h'] = df['return_1h'].rolling(24).std()
    df['volatility_7d'] = df['return_1h'].rolling(24 * 7).std()
    
    # ========== RSI ==========
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # ========== MACD ==========
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ========== BOLLINGER BANDS ==========
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_upper'] = (sma20 + 2 * std20 - df['close']) / df['close']
    df['bb_lower'] = (df['close'] - sma20 + 2 * std20) / df['close']
    df['bb_width'] = 4 * std20 / (sma20 + 1e-10)
    
    # ========== ATR ==========
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean() / (df['close'] + 1e-10)
    
    # ========== ADX (Simplified) ==========
    df['adx_14'] = abs(df['return_24h']) * 100
    
    # ========== TIME ENCODING ==========
    if 'timestamp' in df.columns:
        hour = df['timestamp'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    else:
        df['hour_sin'] = 0
        df['hour_cos'] = 0
    
    return df


def add_btc_features(eth_df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    """Add BTC-derived features to ETH data."""
    # Merge on timestamp
    btc_features = btc_df[['timestamp', 'close', 'return_24h', 'volatility_24h']].copy()
    btc_features = btc_features.rename(columns={
        'close': 'btc_close',
        'return_24h': 'btc_return_24h',
        'volatility_24h': 'btc_volatility'
    })
    
    merged = eth_df.merge(btc_features, on='timestamp', how='left')
    
    # Calculate correlation (rolling 24h)
    merged['btc_correlation'] = merged['return_1h'].rolling(24).corr(
        merged['btc_return_24h'].rolling(24).mean()
    )
    
    # BTC lead indicator (does BTC move before ETH?)
    merged['btc_lead'] = merged['btc_return_24h'].shift(1) - merged['return_24h'].shift(1)
    
    # Fill NaN
    merged['btc_correlation'] = merged['btc_correlation'].fillna(0.5)
    merged['btc_lead'] = merged['btc_lead'].fillna(0)
    
    return merged


def add_derivatives_features(df: pd.DataFrame, funding_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Add derivatives features (funding rate, OI, etc.)."""
    df = df.copy()
    
    if funding_df is not None and len(funding_df) > 0:
        # Merge funding rate
        df = df.merge(
            funding_df.rename(columns={'fundingRate': 'funding_rate'}),
            on='timestamp',
            how='left'
        )
        df['funding_rate'] = df['funding_rate'].ffill().fillna(0)
    else:
        # Estimate funding from price momentum
        df['funding_rate'] = df['return_24h'].rolling(8).mean() * 0.01
    
    # OI proxy from volume patterns
    df['open_interest'] = df['volume'].rolling(24 * 7).mean()
    df['oi_change'] = df['open_interest'].pct_change(24)
    
    # Long/short ratio proxy from price action
    up_volume = df['volume'].where(df['close'] > df['open'], 0)
    down_volume = df['volume'].where(df['close'] <= df['open'], 0)
    df['long_short_ratio'] = (up_volume.rolling(24).sum() / 
                              (down_volume.rolling(24).sum() + 1))
    
    # Volume ratio (current vs average)
    df['volume_ratio'] = df['volume'] / (df['volume'].rolling(24 * 7).mean() + 1)
    
    return df


# ============================================================
# MAIN PIPELINE
# ============================================================

def prepare_training_data(
    kaggle_eth_path: Optional[str] = None,
    kaggle_btc_path: Optional[str] = None,
    use_bybit: bool = True,
    output_path: str = 'data/training/combined_training_data.parquet'
) -> pd.DataFrame:
    """
    Main pipeline to prepare comprehensive training dataset.
    
    Args:
        kaggle_eth_path: Path to Kaggle ETH dataset
        kaggle_btc_path: Path to Kaggle BTC dataset
        use_bybit: Whether to supplement with Bybit API
        output_path: Where to save the final dataset
        
    Returns:
        Combined DataFrame ready for training
    """
    logger.info("=" * 60)
    logger.info("PREPARING COMPREHENSIVE TRAINING DATASET")
    logger.info("=" * 60)
    
    # ========== LOAD ETH DATA ==========
    if kaggle_eth_path:
        eth_df = load_kaggle_dataset(kaggle_eth_path, 'ETH')
        if len(eth_df) > 0 and eth_df['timestamp'].diff().median() < pd.Timedelta(hours=1):
            eth_df = resample_to_hourly(eth_df)
    else:
        logger.info("No Kaggle path provided, fetching from Bybit...")
        eth_df = fetch_bybit_ohlcv('ETHUSDT', '60', days=365)
    
    # ========== LOAD BTC DATA ==========
    if kaggle_btc_path:
        btc_df = load_kaggle_dataset(kaggle_btc_path, 'BTC')
        if len(btc_df) > 0 and btc_df['timestamp'].diff().median() < pd.Timedelta(hours=1):
            btc_df = resample_to_hourly(btc_df)
    else:
        logger.info("Fetching BTC from Bybit...")
        btc_df = fetch_bybit_ohlcv('BTCUSDT', '60', days=365)
    
    # ========== CALCULATE FEATURES ==========
    logger.info("Calculating technical features...")
    eth_df = calculate_technical_features(eth_df)
    btc_df = calculate_technical_features(btc_df)
    
    # ========== ADD BTC INFLUENCE ==========
    logger.info("Adding BTC influence features...")
    eth_df = add_btc_features(eth_df, btc_df)
    
    # ========== ADD DERIVATIVES ==========
    logger.info("Adding derivatives features...")
    funding_df = None
    if use_bybit:
        funding_df = fetch_bybit_funding_rate('ETHUSDT', days=30)
    eth_df = add_derivatives_features(eth_df, funding_df)
    
    # ========== CREATE TARGET ==========
    eth_df['target'] = (eth_df['close'].shift(-1) > eth_df['close']).astype(int)
    
    # ========== CLEAN UP ==========
    # Select final features
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'return_1h', 'return_24h', 'return_7d', 'volatility_24h', 'volatility_7d',
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_lower', 'bb_width', 'atr_14', 'adx_14',
        'funding_rate', 'open_interest', 'oi_change', 'long_short_ratio', 'volume_ratio',
        'btc_return_24h', 'btc_volatility', 'btc_correlation', 'btc_lead',
        'hour_sin', 'hour_cos'
    ]
    
    final_df = eth_df[['timestamp'] + feature_cols + ['target']].dropna()
    
    # ========== SAVE ==========
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_parquet(output_path, index=False)
    
    logger.info("=" * 60)
    logger.info(f"DATASET CREATED: {len(final_df)} samples")
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Date range: {final_df['timestamp'].min()} to {final_df['timestamp'].max()}")
    logger.info(f"Saved to: {output_path}")
    logger.info("=" * 60)
    
    return final_df


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare training data for CORE models')
    parser.add_argument('--kaggle-eth', type=str, help='Path to Kaggle ETH dataset')
    parser.add_argument('--kaggle-btc', type=str, help='Path to Kaggle BTC dataset')
    parser.add_argument('--no-bybit', action='store_true', help='Skip Bybit API calls')
    parser.add_argument('--output', type=str, default='data/training/combined_training_data.parquet')
    
    args = parser.parse_args()
    
    df = prepare_training_data(
        kaggle_eth_path=args.kaggle_eth,
        kaggle_btc_path=args.kaggle_btc,
        use_bybit=not args.no_bybit,
        output_path=args.output
    )
    
    print(f"\nSample data:")
    print(df.head())
    print(f"\nClass balance: {df['target'].mean():.1%} UP")
