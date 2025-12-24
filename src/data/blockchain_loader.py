#!/usr/bin/env python3
"""
PhiHorizon V6.1 Blockchain.com Data Loader

Loads on-chain data from Blockchain.com API (FREE).

Data provided:
- Hash Rate (network mining power)
- Mining Difficulty
- Block height

API: https://blockchain.info/

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
from typing import Dict, Optional, Tuple, Any

import pandas as pd
import numpy as np
import requests

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass  
class BlockchainConfig:
    """Configuration for Blockchain.com API."""
    
    charts_url: str = "https://api.blockchain.info/charts"
    stats_url: str = "https://api.blockchain.info/stats"
    
    # Rate limiting
    request_delay: float = 1.0
    max_retries: int = 3
    retry_backoff: float = 2.0
    
    # Cache settings
    cache_ttl_hours: int = 6  # On-chain data updates less frequently
    
    # Signal thresholds
    hash_rate_growth_threshold: float = 0.10  # 10% growth = bullish


# ============================================================
# BLOCKCHAIN LOADER
# ============================================================

class BlockchainLoader:
    """
    Loads on-chain data from Blockchain.com.
    
    Provides:
    - Hash Rate (hash_rate)
    - Hash Rate Change (hash_rate_change)
    - Mining Difficulty (difficulty)
    """
    
    def __init__(self, config: Optional[BlockchainConfig] = None):
        """Initialize the BlockchainLoader."""
        self.config = config or BlockchainConfig()
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._last_request_time = 0.0
        
        logger.info("BlockchainLoader initialized")
    
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
        url: str,
        params: Optional[Dict] = None
    ) -> Optional[Any]:
        """
        Make HTTP request with rate limiting and retries.
        
        Args:
            url: Full URL
            params: Optional query parameters
            
        Returns:
            JSON response, or None if failed
        """
        for attempt in range(self.config.max_retries):
            try:
                self._rate_limit()
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                return response.json()
                
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_backoff ** attempt)
        
        logger.error(f"All {self.config.max_retries} attempts failed")
        return None
    
    # ================================================================
    # HASH RATE
    # ================================================================
    
    def get_hash_rate(
        self,
        timespan: str = "1year",
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get historical hash rate data.
        
        Args:
            timespan: Time period (1year, 2years, all)
            use_cache: Whether to use cache
            
        Returns:
            DataFrame with timestamp and hash_rate
        """
        cache_key = f"hash_rate_{timespan}"
        
        if use_cache and self._is_cache_valid(cache_key):
            logger.info("Using cached hash rate data")
            return self._cache[cache_key][1]
        
        logger.info(f"Fetching hash rate ({timespan})...")
        
        url = f"{self.config.charts_url}/hash-rate"
        data = self._make_request(url, {
            "timespan": timespan,
            "format": "json",
            "sampled": "true"
        })
        
        if not data or "values" not in data:
            logger.warning("Could not fetch hash rate, using simulated data")
            return self._generate_simulated_hash_rate()
        
        records = []
        for item in data["values"]:
            records.append({
                "timestamp": datetime.fromtimestamp(item["x"]),
                "hash_rate": item["y"],  # TH/s
            })
        
        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Add derived features
        df = self._add_hash_rate_features(df)
        
        self._cache[cache_key] = (datetime.now(), df)
        
        logger.info(f"Loaded {len(df)} hash rate records")
        
        return df
    
    def _add_hash_rate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to hash rate data."""
        df = df.copy()
        
        # Moving averages
        df['hash_rate_ma7'] = df['hash_rate'].rolling(7).mean()
        df['hash_rate_ma30'] = df['hash_rate'].rolling(30).mean()
        
        # Changes
        df['hash_rate_change_7d'] = df['hash_rate'].pct_change(7)
        df['hash_rate_change_30d'] = df['hash_rate'].pct_change(30)
        
        # Trend
        df['hash_rate_trend'] = (df['hash_rate_ma7'] > df['hash_rate_ma30']).astype(int)
        
        return df
    
    def _generate_simulated_hash_rate(self, days: int = 365) -> pd.DataFrame:
        """Generate simulated hash rate for backtesting."""
        logger.info("Generating simulated hash rate...")
        
        np.random.seed(42)
        
        dates = pd.date_range(
            end=datetime.now(),
            periods=days,
            freq='D'
        )
        
        # Hash rate generally trends up with some volatility
        # Start at 400 EH/s, trend to 600 EH/s
        trend = np.linspace(400e6, 600e6, days)  # In TH/s
        noise = np.random.normal(0, 10e6, days)
        
        # Add occasional dips (China ban, etc.)
        dips = np.zeros(days)
        for i in range(0, days, 90):
            if np.random.random() > 0.7:
                dip_len = np.random.randint(10, 30)
                dip_mag = np.random.uniform(0.1, 0.3)
                for j in range(i, min(i + dip_len, days)):
                    dips[j] = -trend[j] * dip_mag * np.exp(-(j - i) / 10)
        
        hash_rate = trend + noise + dips
        hash_rate = np.maximum(hash_rate, 100e6)  # Min 100 EH/s
        
        df = pd.DataFrame({
            'timestamp': dates,
            'hash_rate': hash_rate,
        })
        
        df = self._add_hash_rate_features(df)
        
        return df
    
    def get_current_hash_rate(self) -> Optional[Dict]:
        """Get current hash rate."""
        df = self.get_hash_rate()
        
        if df is None or df.empty:
            return None
        
        latest = df.iloc[-1]
        
        return {
            'hash_rate': float(latest['hash_rate']),
            'hash_rate_th': float(latest['hash_rate']) / 1e6,  # Convert to EH/s
            'change_7d': float(latest['hash_rate_change_7d']) if pd.notna(latest['hash_rate_change_7d']) else 0,
            'change_30d': float(latest['hash_rate_change_30d']) if pd.notna(latest['hash_rate_change_30d']) else 0,
            'trend': 'up' if latest['hash_rate_trend'] == 1 else 'down',
            'timestamp': latest['timestamp'],
        }
    
    # ================================================================
    # DIFFICULTY
    # ================================================================
    
    def get_difficulty(
        self,
        timespan: str = "1year",
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get historical mining difficulty data.
        
        Args:
            timespan: Time period
            use_cache: Whether to use cache
            
        Returns:
            DataFrame with timestamp and difficulty
        """
        cache_key = f"difficulty_{timespan}"
        
        if use_cache and self._is_cache_valid(cache_key):
            logger.info("Using cached difficulty data")
            return self._cache[cache_key][1]
        
        logger.info(f"Fetching difficulty ({timespan})...")
        
        url = f"{self.config.charts_url}/difficulty"
        data = self._make_request(url, {
            "timespan": timespan,
            "format": "json",
            "sampled": "true"
        })
        
        if not data or "values" not in data:
            logger.warning("Could not fetch difficulty")
            return None
        
        records = []
        for item in data["values"]:
            records.append({
                "timestamp": datetime.fromtimestamp(item["x"]),
                "difficulty": item["y"],
            })
        
        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Changes
        df['difficulty_change'] = df['difficulty'].pct_change()
        
        self._cache[cache_key] = (datetime.now(), df)
        
        logger.info(f"Loaded {len(df)} difficulty records")
        
        return df
    
    # ================================================================
    # NETWORK STATS
    # ================================================================
    
    def get_network_stats(self, use_cache: bool = True) -> Optional[Dict]:
        """
        Get current network statistics.
        
        Returns:
            Dict with various network metrics
        """
        cache_key = "network_stats"
        
        if use_cache and self._is_cache_valid(cache_key):
            return self._cache[cache_key][1]
        
        logger.info("Fetching network stats...")
        
        data = self._make_request(self.config.stats_url)
        
        if not data:
            return None
        
        result = {
            'hash_rate': data.get('hash_rate', 0),
            'difficulty': data.get('difficulty', 0),
            'block_height': data.get('n_blocks_total', 0),
            'market_price_usd': data.get('market_price_usd', 0),
            'n_btc_mined': data.get('n_btc_mined', 0) / 1e8,  # Satoshi to BTC
            'n_tx': data.get('n_tx', 0),
            'timestamp': datetime.now(),
        }
        
        self._cache[cache_key] = (datetime.now(), result)
        
        return result
    
    # ================================================================
    # SIGNAL CLASSIFICATION
    # ================================================================
    
    def classify_hash_rate_trend(self, change_30d: float) -> str:
        """
        Classify hash rate trend.
        
        Args:
            change_30d: 30-day change percentage
            
        Returns:
            "strong_growth", "declining", or "stable"
        """
        if change_30d >= self.config.hash_rate_growth_threshold:
            return "strong_growth"
        elif change_30d <= -self.config.hash_rate_growth_threshold:
            return "declining"
        else:
            return "stable"
    
    def should_allow_trade(self, change_30d: float) -> Tuple[bool, str]:
        """
        Determine if trade should be allowed based on hash rate.
        
        Logic:
        - Strong growth → bullish fundamental
        - Declining → watch for capitulation
        
        Args:
            change_30d: 30-day change percentage
            
        Returns:
            Tuple of (allow_trade, reason)
        """
        classification = self.classify_hash_rate_trend(change_30d)
        
        if classification == "declining":
            return True, f"Declining hash rate ({change_30d:.1%}) - Watch for miner capitulation"
        elif classification == "strong_growth":
            return True, f"Strong hash rate growth ({change_30d:.1%}) - Bullish fundamental"
        else:
            return True, f"Stable hash rate ({change_30d:.1%}) - Neutral"


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_current_hash_rate() -> Optional[float]:
    """Get current Bitcoin hash rate in EH/s."""
    loader = BlockchainLoader()
    data = loader.get_current_hash_rate()
    return data["hash_rate_th"] if data else None


# ============================================================
# MAIN (for testing)
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "=" * 60)
    print("BLOCKCHAIN.COM LOADER TEST")
    print("=" * 60)
    
    loader = BlockchainLoader()
    
    # Test hash rate
    print("\n⛏️ Testing Hash Rate...")
    hr = loader.get_current_hash_rate()
    if hr:
        print(f"✅ Hash Rate: {hr['hash_rate_th']:.2f} EH/s")
        print(f"   7d Change: {hr['change_7d']:.2%}")
        print(f"   30d Change: {hr['change_30d']:.2%}")
        print(f"   Trend: {hr['trend']}")
    
    # Test historical
    print("\n📈 Testing Historical Data...")
    df = loader.get_hash_rate()
    if df is not None:
        print(f"✅ Loaded {len(df)} records")
        print(df.tail(5).to_string(index=False))
    
    # Test network stats
    print("\n🌐 Testing Network Stats...")
    stats = loader.get_network_stats()
    if stats:
        print(f"✅ Block Height: {stats['block_height']:,}")
        print(f"   Difficulty: {stats['difficulty']:,.0f}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
