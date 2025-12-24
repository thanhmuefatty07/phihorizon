#!/usr/bin/env python3
"""
PhiHorizon V6.1 Google Trends Data Loader

Loads search interest data from Google Trends using pytrends.

Data provided:
- Bitcoin search interest (0-100)
- Search interest change (momentum)

Library: pytrends (unofficial Google Trends API)

Quality Checklist:
[x] Docstring for module
[x] Docstring for all functions
[x] Type hints on parameters
[x] Error handling (try/except)
[x] Logging (INFO level)
[x] Rate limiting (careful with Google)
[x] Caching (important - Google blocks frequent requests)

Note: This is for SIMULATION/BACKTESTING only, NOT live trading.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Try to import pytrends
try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    logger.warning("pytrends not installed. Run: pip install pytrends")


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class GoogleTrendsConfig:
    """Configuration for Google Trends."""
    
    # Keywords to track
    keywords: list = None
    
    # Default timeframe
    default_timeframe: str = "today 12-m"  # Last 12 months
    
    # Rate limiting (Google is strict)
    request_delay: float = 5.0  # 5 seconds between requests
    max_retries: int = 3
    
    # Cache settings (longer TTL to reduce Google requests)
    cache_ttl_hours: int = 24  # Cache for 24 hours
    
    # Signal thresholds
    high_interest_threshold: float = 80  # Interest > 80 = FOMO signal
    low_interest_threshold: float = 20   # Interest < 20 = Accumulation
    spike_threshold_pct: float = 0.50    # 50% increase = spike
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = ["Bitcoin", "BTC"]


# ============================================================
# GOOGLE TRENDS LOADER
# ============================================================

class GoogleTrendsLoader:
    """
    Loads search interest data from Google Trends.
    
    Provides:
    - Bitcoin search interest (google_trends)
    - Search momentum (trends_change_7d)
    
    Note: Google Trends data is relative (0-100), not absolute.
    """
    
    def __init__(self, config: Optional[GoogleTrendsConfig] = None):
        """Initialize the GoogleTrendsLoader."""
        self.config = config or GoogleTrendsConfig()
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._last_request_time = 0.0
        self._pytrends = None
        
        if PYTRENDS_AVAILABLE:
            try:
                self._pytrends = TrendReq(hl='en-US', tz=360)
                logger.info("GoogleTrendsLoader initialized")
            except Exception as e:
                logger.error(f"Failed to initialize pytrends: {e}")
        else:
            logger.warning("pytrends not available - using simulated data")
    
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
            sleep_time = self.config.request_delay - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()
    
    # ================================================================
    # SEARCH INTEREST DATA
    # ================================================================
    
    def get_search_interest(
        self,
        keyword: str = "Bitcoin",
        timeframe: str = None,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get historical search interest for a keyword.
        
        Args:
            keyword: Search term (default: "Bitcoin")
            timeframe: Google Trends timeframe format
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with timestamp and interest_score
        """
        timeframe = timeframe or self.config.default_timeframe
        cache_key = f"trends_{keyword}_{timeframe}"
        
        if use_cache and self._is_cache_valid(cache_key):
            logger.info("Using cached Google Trends data")
            return self._cache[cache_key][1]
        
        logger.info(f"Fetching Google Trends for '{keyword}'...")
        
        # Use real pytrends if available
        if self._pytrends:
            df = self._fetch_real_trends(keyword, timeframe)
        else:
            df = self._generate_simulated_trends(keyword)
        
        if df is not None:
            self._cache[cache_key] = (datetime.now(), df)
            logger.info(f"Loaded {len(df)} data points for '{keyword}'")
        
        return df
    
    def _fetch_real_trends(
        self,
        keyword: str,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Fetch real Google Trends data using pytrends."""
        try:
            self._rate_limit()
            
            # Build payload
            self._pytrends.build_payload(
                kw_list=[keyword],
                cat=0,
                timeframe=timeframe,
                geo='',
                gprop=''
            )
            
            # Get interest over time
            df = self._pytrends.interest_over_time()
            
            if df.empty:
                logger.warning(f"No data returned for '{keyword}'")
                return None
            
            # Process data
            df = df.reset_index()
            df = df.rename(columns={
                'date': 'timestamp',
                keyword: 'interest_score'
            })
            
            # Keep only relevant columns
            if 'isPartial' in df.columns:
                df = df.drop(columns=['isPartial'])
            
            df = df[['timestamp', 'interest_score']]
            
            # Add derived features
            df = self._add_derived_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Google Trends: {e}")
            return None
    
    def _generate_simulated_trends(
        self,
        keyword: str,
        days: int = 365
    ) -> pd.DataFrame:
        """
        Generate simulated trends data for backtesting.
        
        Creates realistic patterns:
        - Base level around 50
        - Periodic spikes (FOMO periods)
        - Correlation with market cycles
        """
        logger.info("Generating simulated trends data...")
        
        np.random.seed(42)
        
        dates = pd.date_range(
            end=datetime.now(),
            periods=days,
            freq='D'
        )
        
        # Base interest (random walk around 50)
        base = 50 + np.cumsum(np.random.normal(0, 2, days))
        base = np.clip(base, 10, 90)
        
        # Add periodic spikes
        spikes = np.zeros(days)
        for i in range(0, days, 60):  # Every ~2 months
            if np.random.random() > 0.5:
                spike_len = np.random.randint(5, 15)
                spike_mag = np.random.randint(20, 40)
                spikes[i:min(i+spike_len, days)] = spike_mag * np.exp(-np.arange(min(spike_len, days-i)) / 5)
        
        # Combine
        interest = base + spikes
        interest = np.clip(interest, 0, 100).astype(int)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'interest_score': interest,
        })
        
        df = self._add_derived_features(df)
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to trends data."""
        df = df.copy()
        
        # Moving averages
        df['trends_ma7'] = df['interest_score'].rolling(7).mean()
        df['trends_ma30'] = df['interest_score'].rolling(30).mean()
        
        # Changes
        df['trends_change_1d'] = df['interest_score'].diff(1)
        df['trends_change_7d'] = df['interest_score'].diff(7)
        df['trends_change_pct_7d'] = df['interest_score'].pct_change(7)
        
        # Trend direction
        df['trends_above_ma'] = (df['interest_score'] > df['trends_ma30']).astype(int)
        
        return df
    
    # ================================================================
    # CURRENT DATA
    # ================================================================
    
    def get_current_interest(self, keyword: str = "Bitcoin") -> Optional[Dict]:
        """
        Get current search interest score.
        
        Returns:
            Dict with current interest and classification
        """
        df = self.get_search_interest(keyword)
        
        if df is None or df.empty:
            return None
        
        latest = df.iloc[-1]
        
        result = {
            'interest_score': int(latest['interest_score']),
            'trends_ma7': float(latest['trends_ma7']) if pd.notna(latest['trends_ma7']) else None,
            'trends_change_7d': float(latest['trends_change_7d']) if pd.notna(latest['trends_change_7d']) else None,
            'classification': self.classify_interest(int(latest['interest_score'])),
            'timestamp': latest['timestamp'],
        }
        
        return result
    
    # ================================================================
    # SIGNAL CLASSIFICATION
    # ================================================================
    
    def classify_interest(self, interest_score: int) -> str:
        """
        Classify search interest level.
        
        Args:
            interest_score: Google Trends score (0-100)
            
        Returns:
            "high_fomo", "low_accumulation", or "neutral"
        """
        if interest_score >= self.config.high_interest_threshold:
            return "high_fomo"
        elif interest_score <= self.config.low_interest_threshold:
            return "low_accumulation"
        else:
            return "neutral"
    
    def detect_spike(
        self,
        current: float,
        previous: float
    ) -> Tuple[bool, str]:
        """
        Detect if there's a spike in search interest.
        
        Args:
            current: Current interest score
            previous: Previous period score
            
        Returns:
            Tuple of (is_spike, direction)
        """
        if previous == 0:
            return False, "neutral"
        
        change_pct = (current - previous) / previous
        
        if change_pct >= self.config.spike_threshold_pct:
            return True, "spike_up"
        elif change_pct <= -self.config.spike_threshold_pct:
            return True, "spike_down"
        else:
            return False, "neutral"
    
    def should_allow_trade(self, interest_score: int) -> Tuple[bool, str]:
        """
        Determine if trade should be allowed based on search interest.
        
        Contrarian logic:
        - High FOMO (>80) → block (retail top signal)
        - Low interest (<20) → allow (accumulation)
        
        Args:
            interest_score: Google Trends score
            
        Returns:
            Tuple of (allow_trade, reason)
        """
        classification = self.classify_interest(interest_score)
        
        if classification == "high_fomo":
            return False, f"High FOMO ({interest_score}) - Retail top signal"
        elif classification == "low_accumulation":
            return True, f"Low interest ({interest_score}) - Accumulation phase"
        else:
            return True, f"Neutral interest ({interest_score}) - Defer to other filters"


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_current_bitcoin_trends() -> Optional[int]:
    """Get current Bitcoin search interest."""
    loader = GoogleTrendsLoader()
    data = loader.get_current_interest("Bitcoin")
    return data["interest_score"] if data else None


def is_retail_fomo() -> bool:
    """Check if search interest indicates retail FOMO."""
    interest = get_current_bitcoin_trends()
    if interest is None:
        return False
    loader = GoogleTrendsLoader()
    return loader.classify_interest(interest) == "high_fomo"


# ============================================================
# MAIN (for testing)
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "=" * 60)
    print("GOOGLE TRENDS LOADER TEST")
    print("=" * 60)
    
    loader = GoogleTrendsLoader()
    
    # Test search interest
    print("\n📊 Testing Search Interest...")
    df = loader.get_search_interest("Bitcoin")
    if df is not None:
        print(f"✅ Loaded {len(df)} data points")
        print(f"\nLatest data:")
        print(df.tail(5).to_string(index=False))
    
    # Test current interest
    print("\n🎯 Testing Current Interest...")
    current = loader.get_current_interest("Bitcoin")
    if current:
        print(f"✅ Current Interest: {current['interest_score']}")
        print(f"   Classification: {current['classification']}")
        print(f"   7-day Change: {current['trends_change_7d']}")
        
        # Test signal
        allow, reason = loader.should_allow_trade(current['interest_score'])
        print(f"\n   Allow Trade: {allow}")
        print(f"   Reason: {reason}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
