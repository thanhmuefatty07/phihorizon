#!/usr/bin/env python3
"""
PhiHorizon - TRUE HYBRID Data Pipeline

Combines:
1. Kaggle Datasets (5+ years historical, verified quality)
2. Bybit API (30 days realtime + derivatives)

This ensures:
- Maximum data quality (Kaggle = community verified)
- Historical depth (5+ years = multiple market cycles)
- Fresh data (Bybit = latest 30 days)
- Real derivatives (funding rate, OI from actual API)
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# KAGGLE DATA LOADERS
# ============================================================

def load_kaggle_btc(path: str) -> pd.DataFrame:
    """
    Load BTC data from Kaggle dataset.
    
    Supports common formats:
    - bitstampUSD_1-min_data.csv (mczielinski/bitcoin-historical-data)
    - BTC_1min.csv (generic format)
    """
    logger.info(f"Loading BTC from Kaggle: {path}")
    
    df = pd.read_csv(path)
    
    # Handle different column formats
    if 'Timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    elif 'timestamp' in df.columns:
        if df['timestamp'].dtype == 'int64':
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'open_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    
    # Standardize columns
    col_map = {
        'Open': 'open', 'High': 'high', 'Low': 'low', 
        'Close': 'close', 'Volume': 'volume',
        'Volume_(BTC)': 'volume', 'Volume_(Currency)': 'quote_volume'
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    
    # Keep only needed columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].dropna()
    df = df.drop_duplicates('timestamp').sort_values('timestamp')
    
    logger.info(f"  Loaded {len(df):,} rows: {df.timestamp.min()} to {df.timestamp.max()}")
    return df


def load_kaggle_eth(path: str) -> pd.DataFrame:
    """Load ETH data from Kaggle dataset."""
    logger.info(f"Loading ETH from Kaggle: {path}")
    
    df = pd.read_csv(path)
    
    # Handle different column formats
    if 'Date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['Date'])
    elif 'timestamp' in df.columns:
        if df['timestamp'].dtype == 'int64':
            # Check if milliseconds or seconds
            if df['timestamp'].iloc[0] > 1e12:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'open_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    
    # Standardize columns
    col_map = {
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume'
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].dropna()
    df = df.drop_duplicates('timestamp').sort_values('timestamp')
    
    logger.info(f"  Loaded {len(df):,} rows: {df.timestamp.min()} to {df.timestamp.max()}")
    return df


def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample minute data to hourly OHLCV."""
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
# BYBIT API LOADER (Realtime + Derivatives)
# ============================================================

def fetch_bybit_recent(symbol: str = 'ETHUSDT', days: int = 30) -> pd.DataFrame:
    """Fetch recent OHLCV from Bybit to supplement Kaggle."""
    try:
        import ccxt
    except ImportError:
        logger.warning("CCXT not installed. Run: pip install ccxt")
        return pd.DataFrame()
    
    logger.info(f"Fetching {symbol} from Bybit (last {days} days)...")
    
    exchange = ccxt.bybit({
        'enableRateLimit': True,
        'options': {'defaultType': 'linear'}
    })
    
    since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
    all_ohlcv = []
    
    while len(all_ohlcv) < days * 24:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, '1h', since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
        except Exception as e:
            logger.warning(f"Error: {e}")
            break
    
    if not all_ohlcv:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates('timestamp').sort_values('timestamp')
    
    logger.info(f"  Fetched {len(df)} rows from Bybit")
    return df


def fetch_bybit_derivatives(symbol: str = 'ETHUSDT') -> dict:
    """Fetch real derivatives data from Bybit (funding rate, OI)."""
    try:
        import ccxt
        exchange = ccxt.bybit({'enableRateLimit': True})
        
        # Get current funding rate
        funding = exchange.fetch_funding_rate(symbol)
        
        # Get open interest
        ticker = exchange.fetch_ticker(symbol)
        oi = float(ticker.get('info', {}).get('openInterest', 0))
        
        return {
            'funding_rate': funding.get('fundingRate', 0),
            'open_interest': oi,
            'last_update': datetime.now()
        }
    except Exception as e:
        logger.warning(f"Failed to fetch derivatives: {e}")
        return {'funding_rate': 0, 'open_interest': 0}


# ============================================================
# HYBRID MERGE LOGIC
# ============================================================

def merge_hybrid(kaggle_df: pd.DataFrame, bybit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Kaggle historical data with Bybit realtime data.
    
    Strategy:
    1. Use Kaggle data as base (verified quality)
    2. Append Bybit data for dates after Kaggle cutoff
    3. For overlapping dates, prefer Bybit (fresher)
    """
    if kaggle_df.empty:
        logger.warning("Kaggle data empty, using Bybit only")
        return bybit_df
    
    if bybit_df.empty:
        logger.warning("Bybit data empty, using Kaggle only")
        return kaggle_df
    
    logger.info("Merging Kaggle + Bybit data...")
    
    # Find Kaggle cutoff date
    kaggle_cutoff = kaggle_df['timestamp'].max()
    logger.info(f"  Kaggle cutoff: {kaggle_cutoff}")
    
    # Get Bybit data after Kaggle cutoff
    bybit_new = bybit_df[bybit_df['timestamp'] > kaggle_cutoff]
    logger.info(f"  Bybit new rows: {len(bybit_new)}")
    
    # Combine
    combined = pd.concat([kaggle_df, bybit_new], ignore_index=True)
    combined = combined.drop_duplicates('timestamp', keep='last')
    combined = combined.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"  Combined: {len(combined):,} rows")
    logger.info(f"  Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
    
    return combined


# ============================================================
# MAIN HYBRID PIPELINE
# ============================================================

def prepare_hybrid_dataset(
    kaggle_btc_path: Optional[str] = None,
    kaggle_eth_path: Optional[str] = None,
    bybit_days: int = 30,
    output_path: str = 'data/training/hybrid_training_data.parquet'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main pipeline for TRUE HYBRID data preparation.
    
    Args:
        kaggle_btc_path: Path to Kaggle BTC dataset
        kaggle_eth_path: Path to Kaggle ETH dataset
        bybit_days: Days to fetch from Bybit (realtime)
        output_path: Where to save final dataset
        
    Returns:
        Tuple of (eth_df, btc_df) with combined data
    """
    logger.info("=" * 60)
    logger.info("TRUE HYBRID DATA PIPELINE")
    logger.info("=" * 60)
    
    # ===== LOAD KAGGLE DATA =====
    btc_kaggle = pd.DataFrame()
    eth_kaggle = pd.DataFrame()
    
    if kaggle_btc_path and os.path.exists(kaggle_btc_path):
        btc_kaggle = load_kaggle_btc(kaggle_btc_path)
        # Resample to hourly if minute data
        if len(btc_kaggle) > 100000:  # Likely minute data
            btc_kaggle = resample_to_hourly(btc_kaggle)
    
    if kaggle_eth_path and os.path.exists(kaggle_eth_path):
        eth_kaggle = load_kaggle_eth(kaggle_eth_path)
        if len(eth_kaggle) > 100000:
            eth_kaggle = resample_to_hourly(eth_kaggle)
    
    # ===== FETCH BYBIT REALTIME =====
    btc_bybit = fetch_bybit_recent('BTCUSDT', bybit_days)
    eth_bybit = fetch_bybit_recent('ETHUSDT', bybit_days)
    
    # ===== MERGE HYBRID =====
    btc_combined = merge_hybrid(btc_kaggle, btc_bybit)
    eth_combined = merge_hybrid(eth_kaggle, eth_bybit)
    
    # ===== SUMMARY =====
    logger.info("=" * 60)
    logger.info("HYBRID DATASET CREATED")
    logger.info(f"  BTC: {len(btc_combined):,} samples")
    logger.info(f"  ETH: {len(eth_combined):,} samples")
    logger.info(f"  Total: {len(btc_combined) + len(eth_combined):,} samples")
    logger.info("=" * 60)
    
    return eth_combined, btc_combined


# ============================================================
# KAGGLE NOTEBOOK HELPER
# ============================================================

KAGGLE_NOTEBOOK_CELL = '''
# ============================================================
# CELL 2: TRUE HYBRID DATA LOADING
# ============================================================
# 
# Strategy:
# 1. Load Kaggle datasets (5+ years historical, verified)
# 2. Fetch Bybit API (30 days realtime + derivatives)
# 3. Merge for comprehensive dataset
#
# REQUIRED: Add Kaggle datasets first!
# - "BTC and ETH 1-min Price History" or similar
# ============================================================

!pip install ccxt yfinance -q

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ===== CHECK FOR KAGGLE DATASETS =====
import os

KAGGLE_INPUT = '/kaggle/input'
btc_path = None
eth_path = None

# Auto-detect datasets
for dataset_dir in os.listdir(KAGGLE_INPUT):
    dataset_path = os.path.join(KAGGLE_INPUT, dataset_dir)
    if os.path.isdir(dataset_path):
        for f in os.listdir(dataset_path):
            if 'btc' in f.lower() or 'bitcoin' in f.lower():
                btc_path = os.path.join(dataset_path, f)
            if 'eth' in f.lower() or 'ethereum' in f.lower():
                eth_path = os.path.join(dataset_path, f)

print(f"Found BTC: {btc_path}")
print(f"Found ETH: {eth_path}")

# ===== LOAD KAGGLE DATA =====
def load_kaggle_data(path):
    if path is None or not os.path.exists(path):
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    
    # Auto-detect timestamp column
    for col in ['timestamp', 'Timestamp', 'Date', 'open_time']:
        if col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                unit = 'ms' if df[col].iloc[0] > 1e12 else 's'
                df['timestamp'] = pd.to_datetime(df[col], unit=unit)
            else:
                df['timestamp'] = pd.to_datetime(df[col])
            break
    
    # Standardize columns
    col_map = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    
    # Resample to hourly if minute data
    if len(df) > 50000:
        df = df.set_index('timestamp')
        df = df.resample('1h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna().reset_index()
    
    return df

btc_kaggle = load_kaggle_data(btc_path)
eth_kaggle = load_kaggle_data(eth_path)

print(f"Kaggle BTC: {len(btc_kaggle)} rows")
print(f"Kaggle ETH: {len(eth_kaggle)} rows")

# ===== FETCH BYBIT REALTIME =====
try:
    import ccxt
    
    def fetch_bybit(symbol, days=30):
        exchange = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'linear'}})
        since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
        data = []
        while len(data) < days * 24:
            batch = exchange.fetch_ohlcv(symbol, '1h', since, limit=1000)
            if not batch:
                break
            data.extend(batch)
            since = batch[-1][0] + 1
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    btc_bybit = fetch_bybit('BTCUSDT', 30)
    eth_bybit = fetch_bybit('ETHUSDT', 30)
    print(f"Bybit BTC: {len(btc_bybit)} rows")
    print(f"Bybit ETH: {len(eth_bybit)} rows")
    
except Exception as e:
    print(f"Bybit failed: {e}")
    btc_bybit = pd.DataFrame()
    eth_bybit = pd.DataFrame()

# ===== MERGE HYBRID =====
def merge_data(kaggle, bybit):
    if kaggle.empty:
        return bybit
    if bybit.empty:
        return kaggle
    cutoff = kaggle['timestamp'].max()
    bybit_new = bybit[bybit['timestamp'] > cutoff]
    combined = pd.concat([kaggle, bybit_new]).drop_duplicates('timestamp').sort_values('timestamp')
    return combined.reset_index(drop=True)

btc_df = merge_data(btc_kaggle, btc_bybit)
eth_df = merge_data(eth_kaggle, eth_bybit)

print(f"\\n✓ Combined BTC: {len(btc_df)} rows")
print(f"✓ Combined ETH: {len(eth_df)} rows")
print(f"  Date range: {eth_df['timestamp'].min()} to {eth_df['timestamp'].max()}")
'''


if __name__ == "__main__":
    print(KAGGLE_NOTEBOOK_CELL)
