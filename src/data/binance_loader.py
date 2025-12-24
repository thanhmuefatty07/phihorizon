#!/usr/bin/env python3
"""
PhiHorizon V6.1 Binance Derivatives Data Loader

Loads derivatives market data from Binance Futures API (FREE).

Data provided:
- Open Interest
- Long/Short Ratio
- Top Trader Positions
- Funding Rate (backup for OKX)

API: https://fapi.binance.com/fapi/v1/

Quality Checklist:
[x] Docstring for module
[x] Docstring for all functions
[x] Type hints on parameters
[x] Error handling (try/except)
[x] Logging (INFO level)
[x] Rate limiting
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
class BinanceConfig:
    """Configuration for Binance Futures API."""
    
    base_url: str = "https://fapi.binance.com/fapi/v1"
    
    # Default symbol
    symbol: str = "BTCUSDT"
    
    # Rate limiting (very generous: 2400 req/min)
    request_delay: float = 0.1
    max_retries: int = 3
    retry_backoff: float = 2.0
    
    # Cache settings
    cache_ttl_hours: int = 1
    
    # Thresholds for signals
    ls_ratio_high: float = 2.0   # >2 = overleveraged long
    ls_ratio_low: float = 0.5    # <0.5 = overleveraged short
    oi_change_threshold: float = 0.10  # 10% OI change = significant


# ============================================================
# BINANCE LOADER
# ============================================================

class BinanceLoader:
    """
    Loads derivatives data from Binance Futures.
    
    Provides:
    - Open Interest (open_interest)
    - Long/Short Ratio (long_short_ratio)
    - Top Trader Long/Short Ratio
    - Funding Rate
    """
    
    def __init__(self, config: Optional[BinanceConfig] = None):
        """Initialize the BinanceLoader."""
        self.config = config or BinanceConfig()
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._last_request_time = 0.0
        
        logger.info(f"BinanceLoader initialized for {self.config.symbol}")
    
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
    ) -> Optional[Any]:
        """
        Make HTTP request with rate limiting and retries.
        
        Args:
            endpoint: API endpoint
            params: Optional query parameters
            
        Returns:
            JSON response, or None if failed
        """
        url = f"{self.config.base_url}/{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                self._rate_limit()
                
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 429:
                    logger.warning("Rate limited, waiting 60s...")
                    time.sleep(60)
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
    # OPEN INTEREST
    # ================================================================
    
    def get_open_interest(self, use_cache: bool = True) -> Optional[Dict]:
        """
        Get current open interest for BTCUSDT.
        
        Returns:
            Dict with open_interest and timestamp
        """
        cache_key = f"open_interest_{self.config.symbol}"
        
        if use_cache and self._is_cache_valid(cache_key):
            logger.info("Using cached open interest")
            return self._cache[cache_key][1]
        
        logger.info(f"Fetching open interest for {self.config.symbol}...")
        
        data = self._make_request("openInterest", {"symbol": self.config.symbol})
        
        if not data:
            return None
        
        result = {
            "symbol": data.get("symbol"),
            "open_interest": float(data.get("openInterest", 0)),
            "timestamp": datetime.now(),
        }
        
        self._cache[cache_key] = (datetime.now(), result)
        
        logger.info(f"Open Interest: {result['open_interest']:,.2f} BTC")
        
        return result
    
    def get_open_interest_historical(
        self,
        days: int = 30,
        period: str = "1h"
    ) -> Optional[pd.DataFrame]:
        """
        Get historical open interest data.
        
        Args:
            days: Number of days to fetch
            period: Interval (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
            
        Returns:
            DataFrame with timestamp and open_interest
        """
        logger.info(f"Fetching historical open interest ({days} days, {period})...")
        
        # Calculate time range
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        data = self._make_request("openInterestHist", {
            "symbol": self.config.symbol,
            "period": period,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 500
        })
        
        if not data:
            return None
        
        records = []
        for item in data:
            records.append({
                "timestamp": datetime.fromtimestamp(item["timestamp"] / 1000),
                "open_interest": float(item["sumOpenInterest"]),
                "open_interest_value": float(item["sumOpenInterestValue"]),
            })
        
        df = pd.DataFrame(records)
        
        if len(df) > 0:
            df = df.sort_values("timestamp").reset_index(drop=True)
            # Calculate OI change
            df["oi_change_pct"] = df["open_interest"].pct_change()
        
        logger.info(f"Loaded {len(df)} open interest records")
        
        return df
    
    # ================================================================
    # LONG/SHORT RATIO
    # ================================================================
    
    def get_long_short_ratio(
        self,
        period: str = "1h",
        limit: int = 1
    ) -> Optional[Dict]:
        """
        Get global long/short ratio.
        
        Returns:
            Dict with long_short_ratio and timestamp
        """
        logger.info("Fetching long/short ratio...")
        
        data = self._make_request("globalLongShortAccountRatio", {
            "symbol": self.config.symbol,
            "period": period,
            "limit": limit
        })
        
        if not data or len(data) == 0:
            return None
        
        latest = data[0]
        
        result = {
            "long_short_ratio": float(latest.get("longShortRatio", 1.0)),
            "long_account_pct": float(latest.get("longAccount", 0.5)),
            "short_account_pct": float(latest.get("shortAccount", 0.5)),
            "timestamp": datetime.fromtimestamp(latest["timestamp"] / 1000),
        }
        
        logger.info(f"Long/Short Ratio: {result['long_short_ratio']:.2f}")
        
        return result
    
    def get_long_short_ratio_historical(
        self,
        days: int = 30,
        period: str = "1h"
    ) -> Optional[pd.DataFrame]:
        """
        Get historical long/short ratio.
        
        Args:
            days: Number of days
            period: Interval
            
        Returns:
            DataFrame with timestamp and long_short_ratio
        """
        logger.info(f"Fetching historical L/S ratio ({days} days)...")
        
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        end_time = int(datetime.now().timestamp() * 1000)
        
        data = self._make_request("globalLongShortAccountRatio", {
            "symbol": self.config.symbol,
            "period": period,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 500
        })
        
        if not data:
            return None
        
        records = []
        for item in data:
            records.append({
                "timestamp": datetime.fromtimestamp(item["timestamp"] / 1000),
                "long_short_ratio": float(item["longShortRatio"]),
                "long_account_pct": float(item["longAccount"]),
                "short_account_pct": float(item["shortAccount"]),
            })
        
        df = pd.DataFrame(records)
        
        if len(df) > 0:
            df = df.sort_values("timestamp").reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} L/S ratio records")
        
        return df
    
    # ================================================================
    # TOP TRADER POSITIONS
    # ================================================================
    
    def get_top_trader_ratio(
        self,
        period: str = "1h",
        limit: int = 1
    ) -> Optional[Dict]:
        """
        Get top trader long/short position ratio.
        
        Top traders often predict direction better.
        
        Returns:
            Dict with top_long_short_ratio
        """
        logger.info("Fetching top trader positions...")
        
        data = self._make_request("topLongShortPositionRatio", {
            "symbol": self.config.symbol,
            "period": period,
            "limit": limit
        })
        
        if not data or len(data) == 0:
            return None
        
        latest = data[0]
        
        result = {
            "top_long_short_ratio": float(latest.get("longShortRatio", 1.0)),
            "top_long_account_pct": float(latest.get("longAccount", 0.5)),
            "top_short_account_pct": float(latest.get("shortAccount", 0.5)),
            "timestamp": datetime.fromtimestamp(latest["timestamp"] / 1000),
        }
        
        logger.info(f"Top Trader L/S: {result['top_long_short_ratio']:.2f}")
        
        return result
    
    # ================================================================
    # FUNDING RATE
    # ================================================================
    
    def get_funding_rate(self, limit: int = 1) -> Optional[Dict]:
        """
        Get current funding rate.
        
        Returns:
            Dict with funding_rate
        """
        logger.info("Fetching funding rate...")
        
        data = self._make_request("fundingRate", {
            "symbol": self.config.symbol,
            "limit": limit
        })
        
        if not data or len(data) == 0:
            return None
        
        latest = data[0]
        
        result = {
            "funding_rate": float(latest.get("fundingRate", 0)),
            "funding_time": datetime.fromtimestamp(latest["fundingTime"] / 1000),
            "timestamp": datetime.now(),
        }
        
        logger.info(f"Funding Rate: {result['funding_rate']:.4%}")
        
        return result
    
    def get_funding_rate_historical(
        self,
        days: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        Get historical funding rates.
        
        Args:
            days: Number of days
            
        Returns:
            DataFrame with timestamp and funding_rate
        """
        logger.info(f"Fetching historical funding rate ({days} days)...")
        
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        end_time = int(datetime.now().timestamp() * 1000)
        
        data = self._make_request("fundingRate", {
            "symbol": self.config.symbol,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000
        })
        
        if not data:
            return None
        
        records = []
        for item in data:
            records.append({
                "timestamp": datetime.fromtimestamp(item["fundingTime"] / 1000),
                "funding_rate": float(item["fundingRate"]),
            })
        
        df = pd.DataFrame(records)
        
        if len(df) > 0:
            df = df.sort_values("timestamp").reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} funding rate records")
        
        return df
    
    # ================================================================
    # SIGNAL CLASSIFICATION
    # ================================================================
    
    def classify_long_short_ratio(self, ratio: float) -> str:
        """
        Classify long/short ratio for trading signal.
        
        Args:
            ratio: Long/short ratio
            
        Returns:
            "overleveraged_long", "overleveraged_short", or "neutral"
        """
        if ratio >= self.config.ls_ratio_high:
            return "overleveraged_long"
        elif ratio <= self.config.ls_ratio_low:
            return "overleveraged_short"
        else:
            return "neutral"
    
    def should_allow_trade(self, ratio: float) -> Tuple[bool, str]:
        """
        Determine if trade should be allowed based on L/S ratio.
        
        Contrarian logic:
        - Overleveraged long → block (correction likely)
        - Overleveraged short → allow (squeeze likely)
        
        Args:
            ratio: Long/short ratio
            
        Returns:
            Tuple of (allow_trade, reason)
        """
        classification = self.classify_long_short_ratio(ratio)
        
        if classification == "overleveraged_long":
            return False, f"Overleveraged long (ratio={ratio:.2f}) - Correction risk"
        elif classification == "overleveraged_short":
            return True, f"Overleveraged short (ratio={ratio:.2f}) - Squeeze likely"
        else:
            return True, f"Neutral ratio ({ratio:.2f}) - Defer to other filters"
    
    # ================================================================
    # COMBINED DATA
    # ================================================================
    
    def get_all_derivatives_data(self) -> Optional[Dict]:
        """
        Get all derivatives data in one call.
        
        Returns:
            Dict with all derivatives metrics
        """
        logger.info("Fetching all derivatives data...")
        
        oi = self.get_open_interest()
        ls = self.get_long_short_ratio()
        top = self.get_top_trader_ratio()
        fr = self.get_funding_rate()
        
        if not oi:
            return None
        
        result = {
            "open_interest": oi["open_interest"] if oi else 0,
            "long_short_ratio": ls["long_short_ratio"] if ls else 1.0,
            "long_account_pct": ls["long_account_pct"] if ls else 0.5,
            "top_long_short_ratio": top["top_long_short_ratio"] if top else 1.0,
            "funding_rate": fr["funding_rate"] if fr else 0,
            "timestamp": datetime.now(),
        }
        
        return result


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_current_open_interest() -> Optional[float]:
    """Get current BTC open interest."""
    loader = BinanceLoader()
    data = loader.get_open_interest()
    return data["open_interest"] if data else None


def get_current_long_short_ratio() -> Optional[float]:
    """Get current long/short ratio."""
    loader = BinanceLoader()
    data = loader.get_long_short_ratio()
    return data["long_short_ratio"] if data else None


# ============================================================
# MAIN (for testing)
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "=" * 60)
    print("BINANCE LOADER TEST")
    print("=" * 60)
    
    loader = BinanceLoader()
    
    # Test open interest
    print("\n📊 Testing Open Interest...")
    oi = loader.get_open_interest()
    if oi:
        print(f"✅ Open Interest: {oi['open_interest']:,.2f} BTC")
    
    # Test long/short ratio
    print("\n📈 Testing Long/Short Ratio...")
    ls = loader.get_long_short_ratio()
    if ls:
        print(f"✅ L/S Ratio: {ls['long_short_ratio']:.2f}")
        print(f"   Long: {ls['long_account_pct']:.1%}, Short: {ls['short_account_pct']:.1%}")
        
        # Test signal
        classification = loader.classify_long_short_ratio(ls['long_short_ratio'])
        allow, reason = loader.should_allow_trade(ls['long_short_ratio'])
        print(f"   Signal: {classification}")
        print(f"   Allow trade: {allow}")
    
    # Test top trader
    print("\n👑 Testing Top Trader Ratio...")
    top = loader.get_top_trader_ratio()
    if top:
        print(f"✅ Top Trader L/S: {top['top_long_short_ratio']:.2f}")
    
    # Test funding rate
    print("\n💵 Testing Funding Rate...")
    fr = loader.get_funding_rate()
    if fr:
        print(f"✅ Funding Rate: {fr['funding_rate']:.4%}")
    
    # Test combined
    print("\n📦 Testing All Derivatives Data...")
    all_data = loader.get_all_derivatives_data()
    if all_data:
        print(f"✅ Combined data retrieved successfully")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
