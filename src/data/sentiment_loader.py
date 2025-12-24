#!/usr/bin/env python3
"""
PhiHorizon V6.0 Sentiment Data Loader

Loads sentiment data from free APIs:
1. Fear & Greed Index (Alternative.me) - FREE, unlimited
2. Google Trends (optional) - FREE

This module provides Filter D data for the trading strategy.

Usage:
    from src.data.sentiment_loader import SentimentLoader
    
    loader = SentimentLoader()
    fear_greed = loader.get_fear_greed_index()
    
Note: This is for SIMULATION/BACKTESTING only, NOT live trading.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class SentimentConfig:
    """Configuration for sentiment data loading."""
    
    # Fear & Greed Index API
    fear_greed_url: str = "https://api.alternative.me/fng/"
    fear_greed_limit: int = 365  # Max days to fetch
    
    # Filter D thresholds (based on research)
    extreme_fear_threshold: int = 25   # Below = Extreme Fear (Bullish)
    extreme_greed_threshold: int = 75  # Above = Extreme Greed (Bearish)
    
    # Rate limiting
    request_delay: float = 0.5  # Seconds between requests
    max_retries: int = 3
    
    # Cache settings
    cache_ttl_hours: int = 24  # Cache validity in hours


# ============================================================
# FEAR & GREED INDEX LOADER
# ============================================================

class SentimentLoader:
    """
    Loads sentiment data from free APIs.
    
    Provides:
    - Fear & Greed Index (daily, 2018-present)
    - Google Trends (optional)
    
    For use with Filter D in the trading strategy.
    """
    
    def __init__(self, config: Optional[SentimentConfig] = None):
        self.config = config or SentimentConfig()
        self._cache: Dict[str, Tuple[datetime, pd.DataFrame]] = {}
        
        logger.info("SentimentLoader initialized")
        logger.info(f"Fear threshold: <{self.config.extreme_fear_threshold}")
        logger.info(f"Greed threshold: >{self.config.extreme_greed_threshold}")
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache:
            return False
        
        cache_time, _ = self._cache[key]
        age = datetime.now() - cache_time
        return age < timedelta(hours=self.config.cache_ttl_hours)
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make HTTP request with retries and rate limiting."""
        for attempt in range(self.config.max_retries):
            try:
                time.sleep(self.config.request_delay)
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"All {self.config.max_retries} attempts failed for {url}")
        return None
    
    def get_fear_greed_index(
        self,
        days: int = 365,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch Fear & Greed Index from Alternative.me API.
        
        Args:
            days: Number of days to fetch (max 365)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with columns: [timestamp, value, classification]
            - value: 0-100 (0=Extreme Fear, 100=Extreme Greed)
            - classification: "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
        """
        cache_key = f"fear_greed_{days}"
        
        # Check cache
        if use_cache and self._is_cache_valid(cache_key):
            logger.info(f"Using cached Fear & Greed data")
            return self._cache[cache_key][1]
        
        # Fetch from API
        logger.info(f"Fetching Fear & Greed Index ({days} days)...")
        
        params = {"limit": min(days, self.config.fear_greed_limit)}
        data = self._make_request(self.config.fear_greed_url, params)
        
        if not data or "data" not in data:
            logger.error("Failed to fetch Fear & Greed Index")
            return None
        
        # Parse response
        records = []
        for item in data["data"]:
            records.append({
                "timestamp": datetime.fromtimestamp(int(item["timestamp"])),
                "value": int(item["value"]),
                "classification": item["value_classification"],
            })
        
        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Cache result
        self._cache[cache_key] = (datetime.now(), df)
        
        logger.info(f"Loaded {len(df)} days of Fear & Greed data")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def get_latest_fear_greed(self) -> Optional[Dict]:
        """
        Get the most recent Fear & Greed value.
        
        Returns:
            Dict with keys: value, classification, timestamp
        """
        df = self.get_fear_greed_index(days=1)
        
        if df is None or df.empty:
            return None
        
        latest = df.iloc[-1]
        return {
            "value": int(latest["value"]),
            "classification": latest["classification"],
            "timestamp": latest["timestamp"],
        }
    
    def classify_fear_greed(self, value: int) -> str:
        """
        Classify Fear & Greed value for Filter D.
        
        Args:
            value: Fear & Greed Index (0-100)
            
        Returns:
            "extreme_fear", "extreme_greed", or "neutral"
        """
        if value <= self.config.extreme_fear_threshold:
            return "extreme_fear"
        elif value >= self.config.extreme_greed_threshold:
            return "extreme_greed"
        else:
            return "neutral"
    
    def should_allow_trade(self, value: int) -> Tuple[bool, str]:
        """
        Filter D decision: Should we allow this trade based on sentiment?
        
        Logic:
        - Extreme Fear (<25): ALLOW (contrarian bullish)
        - Extreme Greed (>75): BLOCK (risk of reversal)
        - Neutral (25-75): ALLOW (use other filters)
        
        Args:
            value: Fear & Greed Index (0-100)
            
        Returns:
            Tuple of (allow_trade: bool, reason: str)
        """
        classification = self.classify_fear_greed(value)
        
        if classification == "extreme_fear":
            return True, f"Extreme Fear ({value}) - Contrarian bullish signal"
        elif classification == "extreme_greed":
            return False, f"Extreme Greed ({value}) - High risk of reversal"
        else:
            return True, f"Neutral sentiment ({value}) - Defer to other filters"
    
    def merge_with_ohlcv(
        self,
        ohlcv_df: pd.DataFrame,
        fear_greed_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Merge Fear & Greed data with OHLCV data.
        
        Args:
            ohlcv_df: DataFrame with 'datetime' column
            fear_greed_df: Fear & Greed DataFrame (or fetches if None)
            
        Returns:
            OHLCV DataFrame with added 'fear_greed' and 'fg_classification' columns
        """
        if fear_greed_df is None:
            fear_greed_df = self.get_fear_greed_index()
        
        if fear_greed_df is None:
            logger.warning("No Fear & Greed data available, adding NaN columns")
            ohlcv_df["fear_greed"] = pd.NA
            ohlcv_df["fg_classification"] = pd.NA
            return ohlcv_df
        
        # Ensure datetime columns
        ohlcv_df = ohlcv_df.copy()
        if "datetime" in ohlcv_df.columns:
            ohlcv_df["date"] = pd.to_datetime(ohlcv_df["datetime"]).dt.date
        else:
            logger.error("OHLCV DataFrame must have 'datetime' column")
            return ohlcv_df
        
        fear_greed_df = fear_greed_df.copy()
        fear_greed_df["date"] = pd.to_datetime(fear_greed_df["timestamp"]).dt.date
        
        # Rename for merge
        fg_for_merge = fear_greed_df[["date", "value", "classification"]].rename(
            columns={"value": "fear_greed", "classification": "fg_classification"}
        )
        
        # Merge on date (Fear & Greed is daily)
        merged = ohlcv_df.merge(fg_for_merge, on="date", how="left")
        
        # Forward fill for missing days (weekends, holidays)
        merged["fear_greed"] = merged["fear_greed"].ffill()
        merged["fg_classification"] = merged["fg_classification"].ffill()
        
        # Drop helper column
        merged = merged.drop(columns=["date"])
        
        logger.info(f"Merged Fear & Greed data: {merged['fear_greed'].notna().sum()} rows with data")
        
        return merged


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_current_fear_greed() -> Optional[int]:
    """Quick function to get current Fear & Greed value."""
    loader = SentimentLoader()
    latest = loader.get_latest_fear_greed()
    return latest["value"] if latest else None


def is_extreme_fear() -> bool:
    """Check if market is in Extreme Fear."""
    value = get_current_fear_greed()
    return value is not None and value <= 25


def is_extreme_greed() -> bool:
    """Check if market is in Extreme Greed."""
    value = get_current_fear_greed()
    return value is not None and value >= 75


# ============================================================
# MAIN (for testing)
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "=" * 60)
    print("SENTIMENT LOADER TEST")
    print("=" * 60)
    
    loader = SentimentLoader()
    
    # Test Fear & Greed Index
    print("\n📊 Testing Fear & Greed Index API...")
    
    # Get latest
    latest = loader.get_latest_fear_greed()
    if latest:
        print(f"\n✅ Latest Fear & Greed:")
        print(f"   Value: {latest['value']}")
        print(f"   Classification: {latest['classification']}")
        print(f"   Timestamp: {latest['timestamp']}")
        
        # Test filter decision
        allow, reason = loader.should_allow_trade(latest['value'])
        print(f"\n🎯 Filter D Decision:")
        print(f"   Allow Trade: {allow}")
        print(f"   Reason: {reason}")
    else:
        print("❌ Failed to fetch Fear & Greed data")
    
    # Get historical
    print("\n📈 Fetching historical data...")
    df = loader.get_fear_greed_index(days=30)
    if df is not None:
        print(f"\n✅ Historical Data (last 30 days):")
        print(df.tail(10).to_string(index=False))
        
        # Stats
        print(f"\n📊 Statistics:")
        print(f"   Mean: {df['value'].mean():.1f}")
        print(f"   Min: {df['value'].min()} ({df.loc[df['value'].idxmin(), 'classification']})")
        print(f"   Max: {df['value'].max()} ({df.loc[df['value'].idxmax(), 'classification']})")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
