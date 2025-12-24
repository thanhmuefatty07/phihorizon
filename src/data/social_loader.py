#!/usr/bin/env python3
"""
PhiHorizon V7.0 Social Media Data Loader

Loads social sentiment data from multiple platforms:
1. LunarCrush API (Free tier) - Social metrics aggregator
2. Twitter/X (via Nitter proxy) - Crypto influencer tweets
3. Telegram (via public channels) - Group sentiment

For CORE 2: NLP processing.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import re

import pandas as pd
import numpy as np
import requests

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class SocialConfig:
    """Configuration for social media data loading."""
    
    # LunarCrush API (Free tier available)
    lunarcrush_url: str = "https://lunarcrush.com/api3"
    lunarcrush_api_key: str = ""  # Set via environment
    
    # Nitter instances (Twitter proxy - no auth needed)
    nitter_instances: List[str] = field(default_factory=lambda: [
        "https://nitter.net",
        "https://nitter.cz",
        "https://nitter.poast.org"
    ])
    
    # Crypto influencers to track
    crypto_influencers: List[str] = field(default_factory=lambda: [
        "whale_alert", "100trillionUSD", "APompliano",
        "CryptoCapo_", "AltcoinGordon", "TheCryptoDog",
        "CryptoCred", "inversebrah", "CryptoHayes"
    ])
    
    # Telegram public channels
    telegram_channels: List[str] = field(default_factory=lambda: [
        "whale_alert_io", "crypto_signals_official"
    ])
    
    # Rate limiting
    request_delay: float = 1.0
    max_retries: int = 3
    cache_ttl_hours: int = 1
    
    # Sentiment scoring weights
    engagement_weight: float = 0.4
    sentiment_weight: float = 0.6
    
    # Thresholds
    min_engagement: int = 100  # Minimum likes/retweets
    influencer_multiplier: float = 2.0  # Weight influencer posts more


# ============================================================
# LUNARCRUSH LOADER
# ============================================================

class LunarCrushLoader:
    """
    Loads social metrics from LunarCrush API.
    
    Provides:
    - Galaxy Score (social activity metric)
    - Social Volume
    - Social Engagement
    - Sentiment Score
    """
    
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.base_url = "https://lunarcrush.com/api3"
        self._cache: Dict[str, Tuple[datetime, any]] = {}
        self._last_request = 0.0
        
    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_request = time.time()
        
    def get_coin_metrics(self, symbol: str = "BTC") -> Optional[Dict]:
        """
        Get social metrics for a coin.
        
        Returns:
            Dict with galaxy_score, social_volume, sentiment, etc.
        """
        if not self.api_key:
            logger.warning("LunarCrush API key not set, using simulated data")
            return self._simulate_metrics()
            
        cache_key = f"lunarcrush_{symbol}"
        if cache_key in self._cache:
            cached_time, data = self._cache[cache_key]
            if datetime.now() - cached_time < timedelta(hours=1):
                return data
                
        self._rate_limit()
        
        try:
            response = requests.get(
                f"{self.base_url}/coins/{symbol}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json().get("data", {})
                result = {
                    "galaxy_score": data.get("galaxy_score", 50),
                    "alt_rank": data.get("alt_rank", 1),
                    "social_volume": data.get("social_volume", 0),
                    "social_score": data.get("social_score", 50),
                    "social_contributors": data.get("social_contributors", 0),
                    "social_engagement": data.get("social_engagement", 0),
                    "average_sentiment": data.get("average_sentiment", 3),
                    "sentiment_absolute": data.get("sentiment_absolute", 0),
                    "sentiment_relative": data.get("sentiment_relative", 0),
                    "news_articles": data.get("news_articles", 0),
                    "timestamp": datetime.now().isoformat()
                }
                self._cache[cache_key] = (datetime.now(), result)
                return result
        except Exception as e:
            logger.error(f"LunarCrush API error: {e}")
            
        return self._simulate_metrics()
    
    def _simulate_metrics(self) -> Dict:
        """Generate simulated metrics for development/testing."""
        return {
            "galaxy_score": np.random.randint(40, 80),
            "alt_rank": 1,
            "social_volume": np.random.randint(10000, 100000),
            "social_score": np.random.randint(40, 80),
            "social_contributors": np.random.randint(1000, 10000),
            "social_engagement": np.random.randint(50000, 500000),
            "average_sentiment": round(np.random.uniform(2.5, 4.0), 2),
            "sentiment_absolute": np.random.randint(1000, 5000),
            "sentiment_relative": round(np.random.uniform(-50, 50), 2),
            "news_articles": np.random.randint(10, 50),
            "timestamp": datetime.now().isoformat(),
            "simulated": True
        }


# ============================================================
# SOCIAL LOADER (Main Class)
# ============================================================

class SocialLoader:
    """
    Comprehensive social media data loader.
    
    Aggregates data from:
    - LunarCrush (social metrics)
    - Nitter (Twitter proxy)
    - Public APIs
    
    Provides:
    - Aggregated social sentiment
    - Influencer analysis
    - Social volume tracking
    """
    
    def __init__(self, config: Optional[SocialConfig] = None):
        """Initialize the Social Loader."""
        self.config = config or SocialConfig()
        self._cache: Dict[str, Tuple[datetime, any]] = {}
        self._last_request_time: float = 0
        
        # Initialize sub-loaders
        self.lunarcrush = LunarCrushLoader(self.config.lunarcrush_api_key)
        
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache:
            return False
        cached_time, _ = self._cache[key]
        ttl = timedelta(hours=self.config.cache_ttl_hours)
        return datetime.now() - cached_time < ttl
        
    def _rate_limit(self):
        """Ensure rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.request_delay:
            time.sleep(self.config.request_delay - elapsed)
        self._last_request_time = time.time()
        
    # ========================================================
    # NITTER (Twitter Proxy)
    # ========================================================
    
    def get_influencer_tweets(
        self,
        username: str,
        count: int = 10
    ) -> List[Dict]:
        """
        Fetch recent tweets from an influencer via Nitter.
        
        Args:
            username: Twitter username
            count: Number of tweets to fetch
            
        Returns:
            List of tweet dictionaries
        """
        cache_key = f"tweets_{username}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key][1]
            
        tweets = []
        
        for instance in self.config.nitter_instances:
            try:
                self._rate_limit()
                url = f"{instance}/{username}/rss"
                
                response = requests.get(
                    url,
                    headers={"User-Agent": "PhiHorizon/7.0"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    # Simple RSS parsing
                    content = response.text
                    
                    # Extract items (simplified)
                    items = re.findall(
                        r'<item>.*?<title>(.*?)</title>.*?<pubDate>(.*?)</pubDate>.*?</item>',
                        content,
                        re.DOTALL
                    )
                    
                    for title, pub_date in items[:count]:
                        tweets.append({
                            "text": self._clean_text(title),
                            "username": username,
                            "source": "twitter",
                            "timestamp": pub_date,
                            "is_influencer": username in self.config.crypto_influencers
                        })
                        
                    if tweets:
                        break
                        
            except Exception as e:
                logger.debug(f"Nitter instance {instance} failed: {e}")
                continue
                
        self._cache[cache_key] = (datetime.now(), tweets)
        return tweets
    
    def _clean_text(self, text: str) -> str:
        """Clean HTML and special characters from text."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove special characters
        text = re.sub(r'&[a-z]+;', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def get_all_influencer_tweets(self) -> List[Dict]:
        """Fetch tweets from all tracked influencers."""
        all_tweets = []
        
        for username in self.config.crypto_influencers[:5]:  # Limit to top 5
            tweets = self.get_influencer_tweets(username)
            all_tweets.extend(tweets)
            
        return all_tweets
    
    # ========================================================
    # AGGREGATED SENTIMENT
    # ========================================================
    
    def calculate_social_sentiment(self) -> Dict:
        """
        Calculate aggregated social sentiment score.
        
        Combines:
        - LunarCrush metrics
        - Influencer tweet sentiment
        
        Returns:
            Dict with sentiment score and components
        """
        cache_key = "aggregated_social"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key][1]
            
        # Get LunarCrush metrics
        lc_metrics = self.lunarcrush.get_coin_metrics("BTC")
        
        # Get influencer tweets
        tweets = self.get_all_influencer_tweets()
        
        # LunarCrush sentiment (1-5 scale → 0-1)
        lc_sentiment = (lc_metrics.get("average_sentiment", 3) - 1) / 4
        
        # Galaxy score (0-100 → 0-1)
        galaxy_normalized = lc_metrics.get("galaxy_score", 50) / 100
        
        # Simple tweet sentiment (keyword-based)
        tweet_sentiments = []
        bullish_keywords = ["bullish", "moon", "pump", "buy", "rally", "ath"]
        bearish_keywords = ["bearish", "crash", "dump", "sell", "drop", "rekt"]
        
        for tweet in tweets:
            text_lower = tweet["text"].lower()
            bullish_count = sum(1 for kw in bullish_keywords if kw in text_lower)
            bearish_count = sum(1 for kw in bearish_keywords if kw in text_lower)
            
            if bullish_count + bearish_count > 0:
                score = bullish_count / (bullish_count + bearish_count)
            else:
                score = 0.5
                
            # Weight influencer tweets more
            if tweet.get("is_influencer"):
                weight = self.config.influencer_multiplier
            else:
                weight = 1.0
                
            tweet_sentiments.append((score, weight))
            
        # Weighted average of tweet sentiments
        if tweet_sentiments:
            total_weight = sum(w for _, w in tweet_sentiments)
            tweet_sentiment = sum(s * w for s, w in tweet_sentiments) / total_weight
        else:
            tweet_sentiment = 0.5
            
        # Combine all sources
        final_sentiment = (
            0.4 * lc_sentiment +
            0.3 * galaxy_normalized +
            0.3 * tweet_sentiment
        )
        
        result = {
            "overall_sentiment": final_sentiment,
            "lunarcrush_sentiment": lc_sentiment,
            "galaxy_score": lc_metrics.get("galaxy_score", 50),
            "social_volume": lc_metrics.get("social_volume", 0),
            "tweet_sentiment": tweet_sentiment,
            "influencer_tweets": len([t for t in tweets if t.get("is_influencer")]),
            "total_tweets": len(tweets),
            "classification": self._classify_sentiment(final_sentiment),
            "timestamp": datetime.now().isoformat()
        }
        
        self._cache[cache_key] = (datetime.now(), result)
        return result
    
    def _classify_sentiment(self, score: float) -> str:
        """Classify sentiment score."""
        if score >= 0.65:
            return "bullish"
        elif score <= 0.35:
            return "bearish"
        else:
            return "neutral"
    
    # ========================================================
    # DATAFRAME OUTPUT
    # ========================================================
    
    def get_social_features(self) -> Dict[str, float]:
        """
        Get social features for model input.
        
        Returns:
            Dict of feature name to value
        """
        sentiment = self.calculate_social_sentiment()
        lc_metrics = self.lunarcrush.get_coin_metrics("BTC")
        
        return {
            "social_sentiment": sentiment["overall_sentiment"],
            "galaxy_score": lc_metrics.get("galaxy_score", 50) / 100,
            "social_volume_norm": min(lc_metrics.get("social_volume", 0) / 100000, 1.0),
            "social_engagement_norm": min(lc_metrics.get("social_engagement", 0) / 500000, 1.0),
            "influencer_sentiment": sentiment["tweet_sentiment"],
            "sentiment_relative": lc_metrics.get("sentiment_relative", 0) / 100
        }
    
    def get_social_dataframe(self, days: int = 7) -> pd.DataFrame:
        """
        Get historical social data as DataFrame.
        
        Note: For historical data, would need to store/fetch from database.
        Currently returns current snapshot.
        """
        features = self.get_social_features()
        
        df = pd.DataFrame([{
            "timestamp": datetime.now(),
            **features
        }])
        
        return df


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_current_social_sentiment() -> Dict:
    """Quick function to get current social sentiment."""
    loader = SocialLoader()
    return loader.calculate_social_sentiment()


def get_social_features() -> Dict[str, float]:
    """Get social features for model input."""
    loader = SocialLoader()
    return loader.get_social_features()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    loader = SocialLoader()
    
    print("Testing Social Loader...")
    
    # LunarCrush metrics
    print("\n=== LunarCrush Metrics ===")
    metrics = loader.lunarcrush.get_coin_metrics("BTC")
    print(json.dumps(metrics, indent=2))
    
    # Aggregated sentiment
    print("\n=== Aggregated Social Sentiment ===")
    sentiment = loader.calculate_social_sentiment()
    print(json.dumps(sentiment, indent=2))
    
    # Features for model
    print("\n=== Social Features for Model ===")
    features = loader.get_social_features()
    for k, v in features.items():
        print(f"  {k}: {v:.4f}")
