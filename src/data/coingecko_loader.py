#!/usr/bin/env python3
"""
PhiHorizon V6.1 CoinGecko Data Loader

Loads macro market data from CoinGecko API (FREE).

Data provided:
- BTC Dominance
- Total Market Cap
- Stablecoin Market Cap (USDT + USDC + DAI)

API: https://api.coingecko.com/api/v3/

Quality Checklist:
[x] Docstring for module
[x] Docstring for all functions
[x] Type hints on parameters
[x] Error handling (try/except)
[x] Logging (INFO level)
[x] Rate limiting (30 calls/min free tier)
[x] Caching

Note: This is for SIMULATION/BACKTESTING only, NOT live trading.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import requests

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class CoinGeckoConfig:
    """Configuration for CoinGecko API."""
    
    base_url: str = "https://api.coingecko.com/api/v3"
    
    # Rate limiting (free tier: 10-30 calls/minute)
    request_delay: float = 2.5  # Safe delay between requests
    max_retries: int = 3
    retry_backoff: float = 2.0
    
    # Cache settings
    cache_ttl_hours: int = 1  # Market data updates frequently
    
    # Stablecoins to track
    stablecoin_ids: List[str] = None
    
    def __post_init__(self):
        if self.stablecoin_ids is None:
            self.stablecoin_ids = ['tether', 'usd-coin', 'dai']


# ============================================================
# COINGECKO LOADER
# ============================================================

class CoinGeckoLoader:
    """
    Loads macro market data from CoinGecko.
    
    Free tier limits: 10-30 calls/min
    
    Provides:
    - BTC Dominance (btc_dominance)
    - Total Market Cap (total_mcap)
    - Stablecoin Market Cap (stablecoin_mcap)
    """
    
    def __init__(self, config: Optional[CoinGeckoConfig] = None):
        """Initialize the CoinGeckoLoader."""
        self.config = config or CoinGeckoConfig()
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._last_request_time = 0.0
        
        logger.info("CoinGeckoLoader initialized")
        logger.info(f"Stablecoins tracked: {self.config.stablecoin_ids}")
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache:
            return False
        
        cache_time, _ = self._cache[key]
        age = datetime.now() - cache_time
        return age < timedelta(hours=self.config.cache_ttl_hours)
    
    def _rate_limit(self):
        """Ensure rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.request_delay:
            time.sleep(self.config.request_delay - elapsed)
        self._last_request_time = time.time()
    
    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Make HTTP request with rate limiting and retries.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Optional query parameters
            
        Returns:
            JSON response dict, or None if failed
        """
        url = f"{self.config.base_url}/{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                self._rate_limit()
                
                response = requests.get(url, params=params, timeout=30)
                
                # Handle rate limiting
                if response.status_code == 429:
                    wait_time = 60  # Wait 1 minute if rate limited
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_backoff ** attempt)
        
        logger.error(f"All {self.config.max_retries} attempts failed for {endpoint}")
        return None
    
    # ================================================================
    # GLOBAL MARKET DATA
    # ================================================================
    
    def get_global_data(self, use_cache: bool = True) -> Optional[Dict]:
        """
        Get global market data including BTC dominance.
        
        Returns:
            Dict with keys: btc_dominance, total_mcap, total_volume
        """
        cache_key = "global_data"
        
        if use_cache and self._is_cache_valid(cache_key):
            logger.info("Using cached global data")
            return self._cache[cache_key][1]
        
        logger.info("Fetching global market data...")
        
        data = self._make_request("global")
        
        if not data or "data" not in data:
            logger.error("Failed to fetch global data")
            return None
        
        global_data = data["data"]
        
        result = {
            "btc_dominance": global_data.get("market_cap_percentage", {}).get("btc", 0),
            "eth_dominance": global_data.get("market_cap_percentage", {}).get("eth", 0),
            "total_mcap_usd": global_data.get("total_market_cap", {}).get("usd", 0),
            "total_volume_usd": global_data.get("total_volume", {}).get("usd", 0),
            "active_cryptocurrencies": global_data.get("active_cryptocurrencies", 0),
            "timestamp": datetime.now(),
        }
        
        self._cache[cache_key] = (datetime.now(), result)
        
        logger.info(f"BTC Dominance: {result['btc_dominance']:.2f}%")
        logger.info(f"Total MCap: ${result['total_mcap_usd']:,.0f}")
        
        return result
    
    def get_btc_dominance(self) -> Optional[float]:
        """Get current BTC dominance percentage."""
        data = self.get_global_data()
        return data["btc_dominance"] if data else None
    
    def get_total_market_cap(self) -> Optional[float]:
        """Get total crypto market cap in USD."""
        data = self.get_global_data()
        return data["total_mcap_usd"] if data else None
    
    # ================================================================
    # STABLECOIN DATA
    # ================================================================
    
    def get_stablecoin_mcap(self, use_cache: bool = True) -> Optional[Dict]:
        """
        Get total stablecoin market cap (USDT + USDC + DAI).
        
        Returns:
            Dict with individual and total stablecoin market caps
        """
        cache_key = "stablecoin_mcap"
        
        if use_cache and self._is_cache_valid(cache_key):
            logger.info("Using cached stablecoin data")
            return self._cache[cache_key][1]
        
        logger.info("Fetching stablecoin market caps...")
        
        ids = ",".join(self.config.stablecoin_ids)
        data = self._make_request("simple/price", {
            "ids": ids,
            "vs_currencies": "usd",
            "include_market_cap": "true"
        })
        
        if not data:
            logger.error("Failed to fetch stablecoin data")
            return None
        
        result = {
            "individual": {},
            "total_mcap_usd": 0,
            "timestamp": datetime.now(),
        }
        
        for coin_id in self.config.stablecoin_ids:
            if coin_id in data:
                mcap = data[coin_id].get("usd_market_cap", 0)
                result["individual"][coin_id] = mcap
                result["total_mcap_usd"] += mcap
        
        self._cache[cache_key] = (datetime.now(), result)
        
        logger.info(f"Stablecoin MCap: ${result['total_mcap_usd']:,.0f}")
        
        return result
    
    def get_total_stablecoin_mcap(self) -> Optional[float]:
        """Get total stablecoin market cap in USD."""
        data = self.get_stablecoin_mcap()
        return data["total_mcap_usd"] if data else None
    
    # ================================================================
    # HISTORICAL DATA
    # ================================================================
    
    def get_btc_dominance_historical(
        self,
        days: int = 365,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get historical BTC dominance data.
        
        Note: CoinGecko free tier doesn't provide historical dominance directly.
        We use market chart data as a proxy.
        
        Args:
            days: Number of days to fetch
            use_cache: Whether to use cache
            
        Returns:
            DataFrame with timestamp from global API (limited history)
        """
        cache_key = f"btc_dominance_hist_{days}"
        
        if use_cache and self._is_cache_valid(cache_key):
            return self._cache[cache_key][1]
        
        logger.info(f"Fetching BTC dominance chart ({days} days)...")
        
        # Get BTC market chart for historical data proxy
        data = self._make_request("coins/bitcoin/market_chart", {
            "vs_currency": "usd",
            "days": str(days),
            "interval": "daily"
        })
        
        if not data or "market_caps" not in data:
            logger.warning("Could not fetch historical dominance")
            return None
        
        # Parse market caps (as proxy for dominance trend)
        records = []
        for item in data["market_caps"]:
            records.append({
                "timestamp": datetime.fromtimestamp(item[0] / 1000),
                "btc_mcap_usd": item[1],
            })
        
        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Note: This is BTC market cap, not dominance %
        # For true dominance, we'd need total market cap history too
        
        self._cache[cache_key] = (datetime.now(), df)
        
        logger.info(f"Loaded {len(df)} days of BTC market cap data")
        
        return df
    
    # ================================================================
    # COMBINED DATA
    # ================================================================
    
    def get_all_macro_data(self) -> Optional[Dict]:
        """
        Get all macro data in one call (efficient).
        
        Returns:
            Dict with btc_dominance, total_mcap, stablecoin_mcap
        """
        logger.info("Fetching all macro data...")
        
        global_data = self.get_global_data()
        stablecoin_data = self.get_stablecoin_mcap()
        
        if not global_data:
            return None
        
        result = {
            "btc_dominance": global_data["btc_dominance"],
            "eth_dominance": global_data["eth_dominance"],
            "total_mcap_usd": global_data["total_mcap_usd"],
            "total_volume_usd": global_data["total_volume_usd"],
            "stablecoin_mcap_usd": stablecoin_data["total_mcap_usd"] if stablecoin_data else 0,
            "timestamp": datetime.now(),
        }
        
        return result


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_current_btc_dominance() -> Optional[float]:
    """Quick function to get current BTC dominance."""
    loader = CoinGeckoLoader()
    return loader.get_btc_dominance()


def get_current_stablecoin_mcap() -> Optional[float]:
    """Quick function to get current stablecoin market cap."""
    loader = CoinGeckoLoader()
    return loader.get_total_stablecoin_mcap()


# ============================================================
# MAIN (for testing)
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "=" * 60)
    print("COINGECKO LOADER TEST")
    print("=" * 60)
    
    loader = CoinGeckoLoader()
    
    # Test global data
    print("\n📊 Testing Global Data...")
    global_data = loader.get_global_data()
    if global_data:
        print(f"✅ BTC Dominance: {global_data['btc_dominance']:.2f}%")
        print(f"   Total MCap: ${global_data['total_mcap_usd']:,.0f}")
    
    # Test stablecoin data
    print("\n💵 Testing Stablecoin Data...")
    stable_data = loader.get_stablecoin_mcap()
    if stable_data:
        print(f"✅ Stablecoin MCap: ${stable_data['total_mcap_usd']:,.0f}")
        for coin, mcap in stable_data['individual'].items():
            print(f"   {coin}: ${mcap:,.0f}")
    
    # Test combined
    print("\n📈 Testing All Macro Data...")
    all_data = loader.get_all_macro_data()
    if all_data:
        print(f"✅ Combined data retrieved successfully")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
