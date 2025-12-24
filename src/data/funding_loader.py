#!/usr/bin/env python3
"""
PhiHorizon V6.0 Funding Rate Loader

Loads perpetual funding rate data for Filter E.

Data Source:
- OKX API (Free) - Funding rate history
- Binance API (Free) - Alternative source

Funding Rate Interpretation:
- Positive (>0.01%): Market overleveraged LONG → bearish, likely correction
- Negative (<-0.01%): Market overleveraged SHORT → bullish, short squeeze likely
- Neutral: No strong signal

This module provides Filter E data for the trading strategy.

Research Basis:
- deep_research_v55_parameters.md
- comprehensive_data_sources.md

Note: This is for SIMULATION/BACKTESTING only, NOT live trading.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class FundingConfig:
    """
    Configuration for funding rate loading.
    
    Thresholds based on research:
    - Positive funding >0.01% = overleveraged long
    - Negative funding <-0.01% = overleveraged short
    """
    
    # OKX API (Primary - Free)
    okx_base_url: str = "https://www.okx.com/api/v5/public"
    
    # Binance API (Backup - Free)
    binance_base_url: str = "https://fapi.binance.com/fapi/v1"
    
    # Filter E thresholds (research-based)
    # 0.01% = 0.0001 in decimal
    positive_threshold: float = 0.0001   # >0.01% = overleveraged long
    negative_threshold: float = -0.0001  # <-0.01% = overleveraged short
    
    # Extreme thresholds for stronger signals
    extreme_positive: float = 0.001   # >0.1% = very overleveraged
    extreme_negative: float = -0.001  # <-0.1% = very oversold
    
    # Rate limiting
    request_delay: float = 0.2  # OKX allows faster
    max_retries: int = 3
    
    # Cache settings
    cache_ttl_hours: int = 1  # Funding updates every 8h, cache for 1h
    
    # Instrument
    default_symbol: str = "BTC-USDT-SWAP"  # OKX format
    binance_symbol: str = "BTCUSDT"  # Binance format


# ============================================================
# FUNDING RATE LOADER
# ============================================================

class FundingLoader:
    """
    Loads funding rate data from exchange APIs.
    
    Provides Filter E logic for the trading strategy.
    
    Quality Checklist:
    [x] Docstring for module
    [x] Docstring for all functions
    [x] Type hints on all parameters
    [x] Error handling (try/except)
    [x] Logging (INFO level minimum)
    [x] Rate limiting
    [x] Caching
    """
    
    def __init__(self, config: Optional[FundingConfig] = None):
        """
        Initialize the FundingLoader.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or FundingConfig()
        self._cache: Dict[str, Tuple[datetime, pd.DataFrame]] = {}
        
        logger.info("FundingLoader initialized")
        logger.info(f"Positive threshold: >{self.config.positive_threshold:.4%}")
        logger.info(f"Negative threshold: <{self.config.negative_threshold:.4%}")
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache:
            return False
        
        cache_time, _ = self._cache[key]
        age = datetime.now() - cache_time
        return age < timedelta(hours=self.config.cache_ttl_hours)
    
    def _make_request(
        self,
        url: str,
        params: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Make HTTP request with retries and rate limiting.
        
        Args:
            url: URL to request
            params: Optional query parameters
            
        Returns:
            JSON response dict, or None if failed
        """
        for attempt in range(self.config.max_retries):
            try:
                time.sleep(self.config.request_delay)
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                return response.json()
                
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)
        
        logger.error(f"All {self.config.max_retries} attempts failed for {url}")
        return None
    
    # ================================================================
    # FUNDING RATE FETCHING
    # ================================================================
    
    def get_funding_rate(
        self,
        days: int = 30,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get historical funding rate data.
        
        Args:
            days: Number of days to fetch
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with columns: [timestamp, funding_rate]
        """
        cache_key = f"funding_rate_{days}"
        
        if use_cache and self._is_cache_valid(cache_key):
            logger.info("Using cached funding rate data")
            return self._cache[cache_key][1]
        
        logger.info(f"Fetching funding rate ({days} days)...")
        
        # Try OKX first
        df = self._fetch_okx_funding(days)
        
        # Fallback to Binance
        if df is None:
            logger.info("OKX unavailable, trying Binance...")
            df = self._fetch_binance_funding(days)
        
        # Final fallback to simulated data
        if df is None:
            logger.warning("APIs unavailable, using simulated data")
            df = self._generate_simulated_funding(days)
        
        if df is not None:
            self._cache[cache_key] = (datetime.now(), df)
            logger.info(f"Loaded {len(df)} funding rate records")
        
        return df
    
    def _fetch_okx_funding(self, days: int) -> Optional[pd.DataFrame]:
        """
        Fetch funding rate from OKX API.
        
        Args:
            days: Number of days to fetch
            
        Returns:
            DataFrame or None
        """
        try:
            url = f"{self.config.okx_base_url}/funding-rate-history"
            
            params = {
                'instId': self.config.default_symbol,
                'limit': min(days * 3, 100),  # 3 funding per day
            }
            
            data = self._make_request(url, params)
            
            if not data or 'data' not in data:
                return None
            
            records = []
            for item in data['data']:
                records.append({
                    'timestamp': datetime.fromtimestamp(int(item['fundingTime']) / 1000),
                    'funding_rate': float(item['fundingRate']),
                })
            
            if not records:
                return None
            
            df = pd.DataFrame(records)
            return df.sort_values('timestamp').reset_index(drop=True)
            
        except Exception as e:
            logger.warning(f"OKX API error: {e}")
            return None
    
    def _fetch_binance_funding(self, days: int) -> Optional[pd.DataFrame]:
        """
        Fetch funding rate from Binance API.
        
        Args:
            days: Number of days to fetch
            
        Returns:
            DataFrame or None
        """
        try:
            url = f"{self.config.binance_base_url}/fundingRate"
            
            params = {
                'symbol': self.config.binance_symbol,
                'limit': min(days * 3, 1000),
            }
            
            data = self._make_request(url, params)
            
            if not data:
                return None
            
            records = []
            for item in data:
                records.append({
                    'timestamp': datetime.fromtimestamp(item['fundingTime'] / 1000),
                    'funding_rate': float(item['fundingRate']),
                })
            
            if not records:
                return None
            
            df = pd.DataFrame(records)
            return df.sort_values('timestamp').reset_index(drop=True)
            
        except Exception as e:
            logger.warning(f"Binance API error: {e}")
            return None
    
    def _generate_simulated_funding(self, days: int) -> pd.DataFrame:
        """
        Generate simulated funding rate data for backtesting.
        
        Simulates realistic funding patterns:
        - Mostly near zero
        - Occasional spikes during trends
        
        Args:
            days: Number of days to simulate
            
        Returns:
            DataFrame with simulated funding rates
        """
        np.random.seed(42)
        
        # 3 funding events per day
        n_records = days * 3
        
        dates = pd.date_range(
            end=datetime.now(),
            periods=n_records,
            freq='8H'  # Every 8 hours
        )
        
        # Base funding: slightly positive (normal bull market)
        base_funding = np.random.normal(0.0001, 0.00015, n_records)
        
        # Add trend bias periods
        # Simulate periods of high funding (overleveraged)
        trend_periods = np.zeros(n_records)
        for i in range(0, n_records, 30):  # Every 10 days
            if np.random.random() > 0.7:  # 30% chance of extreme period
                period_length = np.random.randint(3, 10)
                direction = np.random.choice([-1, 1])
                magnitude = np.random.uniform(0.0005, 0.002)
                trend_periods[i:i+period_length] = direction * magnitude
        
        funding_rate = base_funding + trend_periods
        
        df = pd.DataFrame({
            'timestamp': dates,
            'funding_rate': funding_rate,
        })
        
        return df
    
    def get_latest_funding(self) -> Optional[Dict]:
        """
        Get the most recent funding rate.
        
        Returns:
            Dict with keys: funding_rate, timestamp
        """
        df = self.get_funding_rate(days=7)
        
        if df is None or df.empty:
            return None
        
        latest = df.iloc[-1]
        
        return {
            'funding_rate': float(latest['funding_rate']),
            'timestamp': latest['timestamp'],
        }
    
    # ================================================================
    # FILTER E LOGIC
    # ================================================================
    
    def classify_funding_rate(self, funding_rate: float) -> str:
        """
        Classify funding rate for Filter E.
        
        Logic:
        - Positive > threshold: Market overleveraged LONG (bearish)
        - Negative < threshold: Market overleveraged SHORT (bullish)
        - Otherwise: Neutral
        
        Args:
            funding_rate: Funding rate as decimal (e.g., 0.0001 = 0.01%)
            
        Returns:
            "overleveraged_long", "overleveraged_short", or "neutral"
        """
        if funding_rate >= self.config.positive_threshold:
            return "overleveraged_long"
        elif funding_rate <= self.config.negative_threshold:
            return "overleveraged_short"
        else:
            return "neutral"
    
    def should_allow_trade(self, funding_rate: float) -> Tuple[bool, str]:
        """
        Filter E decision: Should we allow this trade based on funding?
        
        Contrarian logic:
        - Overleveraged LONG (positive): BLOCK (likely correction coming)
        - Overleveraged SHORT (negative): ALLOW (short squeeze likely)
        - Neutral: ALLOW (use other filters)
        
        Args:
            funding_rate: Funding rate as decimal
            
        Returns:
            Tuple of (allow_trade: bool, reason: str)
        """
        classification = self.classify_funding_rate(funding_rate)
        
        if classification == "overleveraged_long":
            return False, f"Overleveraged long ({funding_rate:.4%}) - High risk of correction"
        elif classification == "overleveraged_short":
            return True, f"Overleveraged short ({funding_rate:.4%}) - Short squeeze likely"
        else:
            return True, f"Neutral funding ({funding_rate:.4%}) - Defer to other filters"
    
    # ================================================================
    # DATA MERGING
    # ================================================================
    
    def merge_with_ohlcv(
        self,
        ohlcv_df: pd.DataFrame,
        funding_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Merge funding rate data with OHLCV data.
        
        Args:
            ohlcv_df: DataFrame with 'datetime' column
            funding_df: Funding DataFrame (or fetches if None)
            
        Returns:
            OHLCV DataFrame with added 'funding_rate' column
        """
        ohlcv_df = ohlcv_df.copy()
        
        if 'datetime' not in ohlcv_df.columns:
            logger.error("OHLCV DataFrame must have 'datetime' column")
            ohlcv_df['funding_rate'] = pd.NA
            return ohlcv_df
        
        if funding_df is None:
            funding_df = self.get_funding_rate()
        
        if funding_df is None or funding_df.empty:
            logger.warning("No funding data available, adding NaN column")
            ohlcv_df['funding_rate'] = pd.NA
            return ohlcv_df
        
        # Round timestamps to 8-hour periods for matching
        ohlcv_df['funding_period'] = pd.to_datetime(ohlcv_df['datetime']).dt.floor('8H')
        funding_df = funding_df.copy()
        funding_df['funding_period'] = pd.to_datetime(funding_df['timestamp']).dt.floor('8H')
        
        # Merge
        fr_for_merge = funding_df[['funding_period', 'funding_rate']].drop_duplicates('funding_period')
        merged = ohlcv_df.merge(fr_for_merge, on='funding_period', how='left')
        
        # Forward fill
        merged['funding_rate'] = merged['funding_rate'].ffill()
        
        # Drop helper
        merged = merged.drop(columns=['funding_period'])
        
        logger.info(f"Merged funding rate: {merged['funding_rate'].notna().sum()} rows with data")
        
        return merged


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_current_funding() -> Optional[float]:
    """Quick function to get current funding rate."""
    loader = FundingLoader()
    latest = loader.get_latest_funding()
    return latest['funding_rate'] if latest else None


def is_market_overleveraged_long() -> bool:
    """Check if market is overleveraged long."""
    funding = get_current_funding()
    if funding is None:
        return False
    loader = FundingLoader()
    return loader.classify_funding_rate(funding) == "overleveraged_long"


def is_market_overleveraged_short() -> bool:
    """Check if market is overleveraged short."""
    funding = get_current_funding()
    if funding is None:
        return False
    loader = FundingLoader()
    return loader.classify_funding_rate(funding) == "overleveraged_short"


# ============================================================
# MAIN (for testing)
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "=" * 60)
    print("FUNDING RATE LOADER TEST")
    print("=" * 60)
    
    loader = FundingLoader()
    
    # Test funding rate
    print("\n📊 Testing Funding Rate API...")
    
    latest = loader.get_latest_funding()
    if latest:
        print(f"\n✅ Latest Funding Rate:")
        print(f"   Rate: {latest['funding_rate']:.4%}")
        print(f"   Timestamp: {latest['timestamp']}")
        
        # Classify
        classification = loader.classify_funding_rate(latest['funding_rate'])
        print(f"   Classification: {classification}")
        
        # Filter decision
        allow, reason = loader.should_allow_trade(latest['funding_rate'])
        print(f"\n🎯 Filter E Decision:")
        print(f"   Allow Trade: {allow}")
        print(f"   Reason: {reason}")
    else:
        print("❌ Failed to fetch funding data")
    
    # Historical
    print("\n📈 Fetching historical data...")
    df = loader.get_funding_rate(days=7)
    if df is not None:
        print(f"\n✅ Historical Data (last 10 records):")
        print(df.tail(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
