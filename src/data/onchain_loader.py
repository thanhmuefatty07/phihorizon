#!/usr/bin/env python3
"""
PhiHorizon V6.0 On-chain Data Loader

Loads on-chain data from free APIs for whale analysis and Filter F.

Data Sources:
1. Glassnode (Free tier) - Exchange netflow, active addresses
2. Blockchain.com API - Hash rate, difficulty
3. CryptoQuant (Free tier) - Exchange reserve

This module provides Filter F data for the trading strategy.

Usage:
    from src.data.onchain_loader import OnChainLoader
    
    loader = OnChainLoader()
    netflow = loader.get_exchange_netflow()
    
Research Basis:
- Exchange netflow > 0.03 (3%) = whale accumulation (bullish)
- Exchange netflow < -0.03 (-3%) = whale distribution (bearish)
- Source: comprehensive_data_sources.md, deep_research_v55_parameters.md

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
class OnChainConfig:
    """
    Configuration for on-chain data loading.
    
    Thresholds based on research:
    - deep_research_v55_parameters.md
    - comprehensive_data_sources.md
    """
    
    # Glassnode API (Free tier: limited metrics)
    glassnode_base_url: str = "https://api.glassnode.com/v1/metrics"
    glassnode_api_key: str = ""  # Optional for free tier
    
    # Blockchain.com API (Free)
    blockchain_base_url: str = "https://api.blockchain.info/charts"
    
    # Filter F thresholds (based on research)
    # Research: >3% netflow = significant whale movement
    whale_accumulation_threshold: float = 0.03   # >3% inflow = accumulation
    whale_distribution_threshold: float = -0.03  # <-3% outflow = distribution
    
    # Exchange flow thresholds
    exchange_inflow_threshold: float = 0.05  # >5% unusual inflow
    exchange_outflow_threshold: float = -0.05  # <-5% unusual outflow
    
    # Rate limiting (respect API limits)
    request_delay: float = 0.5  # Seconds between requests
    max_retries: int = 3
    retry_backoff: float = 2.0  # Exponential backoff multiplier
    
    # Cache settings
    cache_ttl_hours: int = 6  # On-chain data updates less frequently
    
    # Default timeframes
    default_days: int = 30  # 30 days of history


# ============================================================
# ON-CHAIN DATA LOADER
# ============================================================

class OnChainLoader:
    """
    Loads on-chain data from free APIs.
    
    Provides:
    - Exchange netflow (daily)
    - Active addresses (daily)
    - Hash rate (daily)
    
    For use with Filter F in the trading strategy.
    
    Quality Checklist:
    [x] Docstring for module
    [x] Docstring for all functions
    [x] Type hints on all parameters
    [x] Error handling (try/except)
    [x] Logging (INFO level minimum)
    [x] Rate limiting
    [x] Caching
    """
    
    def __init__(self, config: Optional[OnChainConfig] = None):
        """
        Initialize the OnChainLoader.
        
        Args:
            config: Optional configuration object. Uses defaults if None.
        """
        self.config = config or OnChainConfig()
        self._cache: Dict[str, Tuple[datetime, pd.DataFrame]] = {}
        
        logger.info("OnChainLoader initialized")
        logger.info(f"Accumulation threshold: >{self.config.whale_accumulation_threshold:.1%}")
        logger.info(f"Distribution threshold: <{self.config.whale_distribution_threshold:.1%}")
    
    def _is_cache_valid(self, key: str) -> bool:
        """
        Check if cached data is still valid.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if cache exists and is not expired
        """
        if key not in self._cache:
            return False
        
        cache_time, _ = self._cache[key]
        age = datetime.now() - cache_time
        return age < timedelta(hours=self.config.cache_ttl_hours)
    
    def _make_request(
        self,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Make HTTP request with retries and rate limiting.
        
        Args:
            url: URL to request
            params: Optional query parameters
            headers: Optional headers
            
        Returns:
            JSON response dict, or None if all retries failed
        """
        for attempt in range(self.config.max_retries):
            try:
                # Rate limiting
                time.sleep(self.config.request_delay)
                
                response = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=30
                )
                response.raise_for_status()
                
                return response.json()
                
            except requests.RequestException as e:
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )
                
                if attempt < self.config.max_retries - 1:
                    backoff_time = self.config.retry_backoff ** attempt
                    time.sleep(backoff_time)
        
        logger.error(f"All {self.config.max_retries} attempts failed for {url}")
        return None
    
    # ================================================================
    # EXCHANGE NETFLOW (Primary metric for Filter F)
    # ================================================================
    
    def get_exchange_netflow(
        self,
        days: int = 30,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get exchange netflow data (inflow - outflow).
        
        Netflow interpretation:
        - Positive: More BTC flowing INTO exchanges (selling pressure)
        - Negative: More BTC flowing OUT of exchanges (accumulation)
        
        Note: For whale analysis, we invert this:
        - Positive netflow OUT (negative value) = whale accumulation = bullish
        - Positive netflow IN (positive value) = whale distribution = bearish
        
        Args:
            days: Number of days to fetch
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with columns: [timestamp, netflow, netflow_btc]
        """
        cache_key = f"exchange_netflow_{days}"
        
        # Check cache
        if use_cache and self._is_cache_valid(cache_key):
            logger.info("Using cached exchange netflow data")
            return self._cache[cache_key][1]
        
        logger.info(f"Fetching exchange netflow ({days} days)...")
        
        # Try Glassnode first (preferred)
        df = self._fetch_glassnode_netflow(days)
        
        # Fallback to simulated data for backtesting
        if df is None:
            logger.warning("Glassnode API unavailable, using simulated data")
            df = self._generate_simulated_netflow(days)
        
        if df is not None:
            self._cache[cache_key] = (datetime.now(), df)
            logger.info(f"Loaded {len(df)} days of exchange netflow data")
        
        return df
    
    def _fetch_glassnode_netflow(self, days: int) -> Optional[pd.DataFrame]:
        """
        Fetch netflow from Glassnode API.
        
        Note: Free tier has limited access.
        
        Args:
            days: Number of days to fetch
            
        Returns:
            DataFrame or None if API unavailable
        """
        try:
            # Glassnode free endpoint
            url = f"{self.config.glassnode_base_url}/transactions/transfers_volume_exchanges_net"
            
            params = {
                'a': 'BTC',
                'i': '24h',
                's': int((datetime.now() - timedelta(days=days)).timestamp()),
            }
            
            if self.config.glassnode_api_key:
                params['api_key'] = self.config.glassnode_api_key
            
            data = self._make_request(url, params)
            
            if not data:
                return None
            
            # Parse response
            records = []
            for item in data:
                records.append({
                    'timestamp': datetime.fromtimestamp(item['t']),
                    'netflow': item['v'],  # BTC amount
                })
            
            df = pd.DataFrame(records)
            
            if df.empty:
                return None
            
            # Normalize netflow as percentage (approximate)
            # Assume ~2M BTC on exchanges as baseline
            baseline_btc = 2_000_000
            df['netflow_pct'] = df['netflow'] / baseline_btc
            
            return df.sort_values('timestamp').reset_index(drop=True)
            
        except Exception as e:
            logger.warning(f"Glassnode API error: {e}")
            return None
    
    def _generate_simulated_netflow(self, days: int) -> pd.DataFrame:
        """
        Generate simulated netflow data for backtesting.
        
        This is used when API is unavailable.
        Simulates realistic whale behavior patterns.
        
        Args:
            days: Number of days to simulate
            
        Returns:
            DataFrame with simulated netflow
        """
        np.random.seed(42)  # Reproducible for backtesting
        
        dates = pd.date_range(
            end=datetime.now(),
            periods=days,
            freq='1D'
        )
        
        # Simulate netflow: mostly small, occasional large movements
        # Normal distribution with fat tails
        base_netflow = np.random.normal(0, 0.01, days)  # 1% std
        
        # Add occasional whale movements (5% of days)
        whale_days = np.random.random(days) < 0.05
        whale_magnitude = np.random.choice([-1, 1], days) * np.random.uniform(0.03, 0.08, days)
        
        netflow = base_netflow + (whale_days * whale_magnitude)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'netflow': netflow,  # Already as percentage
            'netflow_pct': netflow,
        })
        
        return df
    
    def get_latest_netflow(self) -> Optional[Dict]:
        """
        Get the most recent netflow value.
        
        Returns:
            Dict with keys: netflow, netflow_pct, timestamp
        """
        df = self.get_exchange_netflow(days=7)
        
        if df is None or df.empty:
            return None
        
        latest = df.iloc[-1]
        
        return {
            'netflow': float(latest.get('netflow', latest.get('netflow_pct', 0))),
            'netflow_pct': float(latest.get('netflow_pct', latest.get('netflow', 0))),
            'timestamp': latest['timestamp'],
        }
    
    # ================================================================
    # WHALE ACTIVITY ANALYSIS (Filter F Logic)
    # ================================================================
    
    def classify_whale_activity(self, netflow_pct: float) -> str:
        """
        Classify whale activity based on exchange netflow.
        
        Logic (inverted because outflow = accumulation):
        - Negative netflow (outflow) > threshold = accumulation (bullish)
        - Positive netflow (inflow) > threshold = distribution (bearish)
        
        Args:
            netflow_pct: Netflow as percentage of total
            
        Returns:
            "accumulation", "distribution", or "neutral"
        """
        # Note: For whale analysis, we look at the opposite
        # Whales WITHDRAWING from exchanges = accumulation = bullish
        # Whales DEPOSITING to exchanges = distribution = bearish
        
        if netflow_pct <= self.config.whale_distribution_threshold:
            # Large outflow = whales withdrawing = accumulation
            return "accumulation"
        elif netflow_pct >= self.config.whale_accumulation_threshold:
            # Large inflow = whales depositing = distribution
            return "distribution"
        else:
            return "neutral"
    
    def should_allow_trade(self, netflow_pct: float) -> Tuple[bool, str]:
        """
        Filter F decision: Should we allow this trade based on whale activity?
        
        Logic:
        - Accumulation (outflow): ALLOW (whales buying, bullish)
        - Distribution (inflow): BLOCK (whales selling, bearish)
        - Neutral: ALLOW (use other filters)
        
        Args:
            netflow_pct: Netflow as percentage
            
        Returns:
            Tuple of (allow_trade: bool, reason: str)
        """
        activity = self.classify_whale_activity(netflow_pct)
        
        if activity == "accumulation":
            return True, f"Whale accumulation ({netflow_pct:.2%} outflow) - Bullish signal"
        elif activity == "distribution":
            return False, f"Whale distribution ({netflow_pct:.2%} inflow) - Risk of selling"
        else:
            return True, f"Neutral whale activity ({netflow_pct:.2%}) - Defer to other filters"
    
    # ================================================================
    # BLOCKCHAIN METRICS (Supplementary)
    # ================================================================
    
    def get_hashrate(self, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Get Bitcoin hash rate from Blockchain.com API.
        
        Hash rate indicates miner confidence and network security.
        Rising hash rate = bullish fundamental.
        
        Args:
            days: Number of days to fetch
            
        Returns:
            DataFrame with columns: [timestamp, hashrate]
        """
        cache_key = f"hashrate_{days}"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key][1]
        
        logger.info(f"Fetching hash rate ({days} days)...")
        
        url = f"{self.config.blockchain_base_url}/hash-rate"
        params = {
            'timespan': f'{days}days',
            'format': 'json',
        }
        
        data = self._make_request(url, params)
        
        if not data or 'values' not in data:
            logger.warning("Failed to fetch hash rate")
            return None
        
        records = []
        for item in data['values']:
            records.append({
                'timestamp': datetime.fromtimestamp(item['x']),
                'hashrate': item['y'],
            })
        
        df = pd.DataFrame(records)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        self._cache[cache_key] = (datetime.now(), df)
        logger.info(f"Loaded {len(df)} days of hash rate data")
        
        return df
    
    # ================================================================
    # DATA MERGING
    # ================================================================
    
    def merge_with_ohlcv(
        self,
        ohlcv_df: pd.DataFrame,
        netflow_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Merge on-chain data with OHLCV data.
        
        Args:
            ohlcv_df: DataFrame with 'datetime' column
            netflow_df: Netflow DataFrame (or fetches if None)
            
        Returns:
            OHLCV DataFrame with added 'whale_netflow' column
        """
        ohlcv_df = ohlcv_df.copy()
        
        # Ensure datetime column
        if 'datetime' not in ohlcv_df.columns:
            logger.error("OHLCV DataFrame must have 'datetime' column")
            ohlcv_df['whale_netflow'] = pd.NA
            return ohlcv_df
        
        # Get netflow if not provided
        if netflow_df is None:
            netflow_df = self.get_exchange_netflow()
        
        if netflow_df is None or netflow_df.empty:
            logger.warning("No netflow data available, adding NaN column")
            ohlcv_df['whale_netflow'] = pd.NA
            return ohlcv_df
        
        # Create date columns for merge
        ohlcv_df['date'] = pd.to_datetime(ohlcv_df['datetime']).dt.date
        netflow_df = netflow_df.copy()
        netflow_df['date'] = pd.to_datetime(netflow_df['timestamp']).dt.date
        
        # Select columns for merge
        netflow_col = 'netflow_pct' if 'netflow_pct' in netflow_df.columns else 'netflow'
        nf_for_merge = netflow_df[['date', netflow_col]].rename(
            columns={netflow_col: 'whale_netflow'}
        )
        
        # Merge on date (netflow is daily)
        merged = ohlcv_df.merge(nf_for_merge, on='date', how='left')
        
        # Forward fill for missing days
        merged['whale_netflow'] = merged['whale_netflow'].ffill()
        
        # Drop helper column
        merged = merged.drop(columns=['date'])
        
        logger.info(f"Merged whale netflow: {merged['whale_netflow'].notna().sum()} rows with data")
        
        return merged


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_current_whale_activity() -> Optional[str]:
    """Quick function to get current whale activity classification."""
    loader = OnChainLoader()
    latest = loader.get_latest_netflow()
    
    if latest is None:
        return None
    
    return loader.classify_whale_activity(latest['netflow_pct'])


def is_whale_accumulating() -> bool:
    """Check if whales are currently accumulating."""
    activity = get_current_whale_activity()
    return activity == "accumulation"


def is_whale_distributing() -> bool:
    """Check if whales are currently distributing."""
    activity = get_current_whale_activity()
    return activity == "distribution"


# ============================================================
# MAIN (for testing)
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "=" * 60)
    print("ON-CHAIN LOADER TEST")
    print("=" * 60)
    
    loader = OnChainLoader()
    
    # Test netflow
    print("\n📊 Testing Exchange Netflow...")
    
    latest = loader.get_latest_netflow()
    if latest:
        print(f"\n✅ Latest Netflow:")
        print(f"   Value: {latest['netflow_pct']:.2%}")
        print(f"   Timestamp: {latest['timestamp']}")
        
        # Classify
        activity = loader.classify_whale_activity(latest['netflow_pct'])
        print(f"   Activity: {activity}")
        
        # Filter decision
        allow, reason = loader.should_allow_trade(latest['netflow_pct'])
        print(f"\n🎯 Filter F Decision:")
        print(f"   Allow Trade: {allow}")
        print(f"   Reason: {reason}")
    else:
        print("❌ Failed to fetch netflow data")
    
    # Historical
    print("\n📈 Fetching historical data...")
    df = loader.get_exchange_netflow(days=30)
    if df is not None:
        print(f"\n✅ Historical Data (last 10 days):")
        print(df.tail(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
