#!/usr/bin/env python3
"""
PhiHorizon V7.0 News Data Loader

Loads news and social sentiment from free APIs:
1. CryptoPanic API (Free tier) - Aggregated crypto news
2. Reddit API (Free) - Crypto subreddit sentiment
3. RSS Feeds - CoinDesk, CoinTelegraph

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
import requests

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class NewsConfig:
    """Configuration for news data loading."""
    
    # CryptoPanic API (Free tier: 5 calls/minute)
    cryptopanic_url: str = "https://cryptopanic.com/api/v1/posts/"
    cryptopanic_auth_token: str = ""  # Set via environment
    
    # Reddit API
    reddit_url: str = "https://www.reddit.com/r/{subreddit}/hot.json"
    reddit_subreddits: List[str] = field(default_factory=lambda: [
        "Bitcoin", "CryptoCurrency", "CryptoMarkets"
    ])
    
    # RSS Feeds (Free, no auth)
    rss_feeds: Dict[str, str] = field(default_factory=lambda: {
        "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "cointelegraph": "https://cointelegraph.com/rss",
        "decrypt": "https://decrypt.co/feed",
    })
    
    # Rate limiting
    request_delay: float = 0.5
    max_retries: int = 3
    cache_ttl_hours: int = 1
    
    # Sentiment keywords
    bullish_keywords: List[str] = field(default_factory=lambda: [
        "bullish", "surge", "rally", "breakout", "all-time high", "ath",
        "moon", "pump", "adoption", "institutional", "buy", "accumulation",
        "etf approved", "halving", "bull run", "recovery"
    ])
    
    bearish_keywords: List[str] = field(default_factory=lambda: [
        "bearish", "crash", "dump", "plunge", "sell-off", "correction",
        "fear", "panic", "ban", "regulation", "hack", "scam", "fraud",
        "sec lawsuit", "bubble", "collapse", "bankruptcy"
    ])


# ============================================================
# NEWS LOADER
# ============================================================

class NewsLoader:
    """
    Loads news data from free APIs for NLP processing.
    
    Provides:
    - Aggregated crypto news from multiple sources
    - Reddit sentiment analysis
    - Simple keyword-based sentiment scoring
    
    For use with CORE 2 NLP in V7.0 architecture.
    """
    
    def __init__(self, config: Optional[NewsConfig] = None):
        """Initialize the NewsLoader."""
        self.config = config or NewsConfig()
        self._cache: Dict[str, Tuple[datetime, any]] = {}
        self._last_request_time: float = 0
        
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
        
    def _make_request(
        self,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Make HTTP request with retries."""
        self._rate_limit()
        
        if headers is None:
            headers = {"User-Agent": "PhiHorizon/7.0"}
            
        for attempt in range(self.config.max_retries):
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=30
                )
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    logger.warning("Rate limited, waiting...")
                    time.sleep(60)
                else:
                    logger.warning(f"Request failed: {response.status_code}")
            except Exception as e:
                logger.error(f"Request error: {e}")
                time.sleep(2 ** attempt)
                
        return None
    
    # ========================================================
    # NEWS FETCHING
    # ========================================================
    
    def get_reddit_posts(
        self,
        subreddit: str = "Bitcoin",
        limit: int = 25
    ) -> List[Dict]:
        """
        Fetch hot posts from a subreddit.
        
        Args:
            subreddit: Subreddit name
            limit: Number of posts to fetch
            
        Returns:
            List of post dictionaries
        """
        cache_key = f"reddit_{subreddit}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key][1]
            
        url = self.config.reddit_url.format(subreddit=subreddit)
        params = {"limit": limit}
        headers = {"User-Agent": "PhiHorizon/7.0 (News Aggregator)"}
        
        data = self._make_request(url, params, headers)
        if not data:
            logger.warning(f"Failed to fetch Reddit data for r/{subreddit}")
            return []
            
        posts = []
        try:
            for child in data.get("data", {}).get("children", []):
                post = child.get("data", {})
                posts.append({
                    "title": post.get("title", ""),
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "created_utc": post.get("created_utc", 0),
                    "subreddit": subreddit,
                    "source": "reddit"
                })
        except Exception as e:
            logger.error(f"Error parsing Reddit data: {e}")
            
        self._cache[cache_key] = (datetime.now(), posts)
        return posts
    
    def get_all_reddit_posts(self, limit: int = 25) -> List[Dict]:
        """Fetch posts from all configured subreddits."""
        all_posts = []
        for subreddit in self.config.reddit_subreddits:
            posts = self.get_reddit_posts(subreddit, limit)
            all_posts.extend(posts)
        return all_posts
    
    def get_cryptopanic_news(self, limit: int = 50) -> List[Dict]:
        """
        Fetch news from CryptoPanic API.
        
        Requires API token in config.
        
        Returns:
            List of news items
        """
        if not self.config.cryptopanic_auth_token:
            logger.warning("CryptoPanic API token not set")
            return []
            
        cache_key = "cryptopanic"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key][1]
            
        params = {
            "auth_token": self.config.cryptopanic_auth_token,
            "currencies": "BTC",
            "kind": "news",
            "public": "true"
        }
        
        data = self._make_request(self.config.cryptopanic_url, params)
        if not data:
            logger.warning("Failed to fetch CryptoPanic news")
            return []
            
        news_items = []
        try:
            for result in data.get("results", [])[:limit]:
                news_items.append({
                    "title": result.get("title", ""),
                    "source": result.get("source", {}).get("title", ""),
                    "published_at": result.get("published_at", ""),
                    "votes": result.get("votes", {}),
                    "url": result.get("url", "")
                })
        except Exception as e:
            logger.error(f"Error parsing CryptoPanic data: {e}")
            
        self._cache[cache_key] = (datetime.now(), news_items)
        return news_items
    
    # ========================================================
    # SENTIMENT ANALYSIS (Simple keyword-based)
    # ========================================================
    
    def analyze_text_sentiment(self, text: str) -> Dict:
        """
        Simple keyword-based sentiment analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with sentiment score and keywords found
        """
        text_lower = text.lower()
        
        bullish_found = [
            kw for kw in self.config.bullish_keywords
            if kw in text_lower
        ]
        bearish_found = [
            kw for kw in self.config.bearish_keywords
            if kw in text_lower
        ]
        
        bullish_count = len(bullish_found)
        bearish_count = len(bearish_found)
        total = bullish_count + bearish_count
        
        if total == 0:
            score = 0.5  # Neutral
        else:
            score = bullish_count / total
            
        return {
            "score": score,  # 0 = bearish, 0.5 = neutral, 1 = bullish
            "bullish_keywords": bullish_found,
            "bearish_keywords": bearish_found,
            "classification": self._classify_score(score)
        }
    
    def _classify_score(self, score: float) -> str:
        """Classify sentiment score."""
        if score >= 0.7:
            return "bullish"
        elif score <= 0.3:
            return "bearish"
        else:
            return "neutral"
    
    def get_aggregated_sentiment(self) -> Dict:
        """
        Get aggregated sentiment from all sources.
        
        Returns:
            Dict with overall sentiment metrics
        """
        cache_key = "aggregated_sentiment"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key][1]
            
        # Collect all text
        all_texts = []
        
        # Reddit posts
        reddit_posts = self.get_all_reddit_posts()
        for post in reddit_posts:
            all_texts.append(post["title"])
            
        # CryptoPanic news
        news_items = self.get_cryptopanic_news()
        for item in news_items:
            all_texts.append(item["title"])
            
        if not all_texts:
            return {
                "score": 0.5,
                "classification": "neutral",
                "sources": 0,
                "timestamp": datetime.now().isoformat()
            }
            
        # Analyze each text
        scores = []
        for text in all_texts:
            result = self.analyze_text_sentiment(text)
            scores.append(result["score"])
            
        avg_score = sum(scores) / len(scores)
        
        result = {
            "score": avg_score,
            "classification": self._classify_score(avg_score),
            "sources": len(all_texts),
            "bullish_count": sum(1 for s in scores if s > 0.6),
            "bearish_count": sum(1 for s in scores if s < 0.4),
            "neutral_count": sum(1 for s in scores if 0.4 <= s <= 0.6),
            "timestamp": datetime.now().isoformat()
        }
        
        self._cache[cache_key] = (datetime.now(), result)
        return result
    
    def get_news_dataframe(self, hours: int = 24) -> pd.DataFrame:
        """
        Get news as DataFrame for NLP processing.
        
        Args:
            hours: How many hours of news to include
            
        Returns:
            DataFrame with news items
        """
        all_items = []
        
        # Reddit
        for post in self.get_all_reddit_posts():
            post_time = datetime.fromtimestamp(post.get("created_utc", 0))
            if datetime.now() - post_time < timedelta(hours=hours):
                all_items.append({
                    "text": post["title"],
                    "source": f"reddit_{post['subreddit']}",
                    "timestamp": post_time,
                    "engagement": post["score"] + post["num_comments"]
                })
                
        # CryptoPanic
        for item in self.get_cryptopanic_news():
            all_items.append({
                "text": item["title"],
                "source": item["source"],
                "timestamp": item.get("published_at", ""),
                "engagement": 0
            })
            
        if not all_items:
            return pd.DataFrame(columns=["text", "source", "timestamp", "engagement"])
            
        return pd.DataFrame(all_items)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_current_sentiment() -> Dict:
    """Quick function to get current news sentiment."""
    loader = NewsLoader()
    return loader.get_aggregated_sentiment()


def get_reddit_sentiment() -> Dict:
    """Get sentiment from Reddit only."""
    loader = NewsLoader()
    posts = loader.get_all_reddit_posts()
    
    if not posts:
        return {"score": 0.5, "classification": "neutral"}
        
    scores = [
        loader.analyze_text_sentiment(p["title"])["score"]
        for p in posts
    ]
    avg = sum(scores) / len(scores)
    
    return {
        "score": avg,
        "classification": loader._classify_score(avg),
        "post_count": len(posts)
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    loader = NewsLoader()
    
    print("Testing Reddit fetch...")
    posts = loader.get_reddit_posts("Bitcoin", limit=5)
    print(f"Got {len(posts)} posts")
    
    if posts:
        print("\nSample post sentiment:")
        sample = posts[0]
        sentiment = loader.analyze_text_sentiment(sample["title"])
        print(f"Title: {sample['title']}")
        print(f"Sentiment: {sentiment}")
    
    print("\nAggregated sentiment:")
    agg = loader.get_aggregated_sentiment()
    print(json.dumps(agg, indent=2))
