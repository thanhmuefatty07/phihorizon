#!/usr/bin/env python3
"""
PhiHorizon V7.0 Data Pipeline

Orchestrates all data sources and feeds to CORE models.

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  QUANT DATA                    NLP DATA                     │
│  ├── Exchange (OHLCV)          ├── News                     │
│  ├── Derivatives (OI, FR)      ├── Reddit                   │
│  ├── On-chain (Whale, Hash)    ├── Twitter/Social           │
│  └── Macro (BTC.D, F&G)        └── LunarCrush               │
│         │                              │                    │
│         ▼                              ▼                    │
│    QuantGuard                     NLPGuard                  │
│         │                              │                    │
│         ▼                              ▼                    │
│     CORE 1                         CORE 2                   │
│   QuantTransformer              NLPFinBERT                  │
│         │                              │                    │
│         └──────────┬───────────────────┘                    │
│                    ▼                                        │
│              FusionGuard                                    │
│                    │                                        │
│                    ▼                                        │
│                 CORE 3                                      │
│           MetaDecisionEngine                                │
│                    │                                        │
│                    ▼                                        │
│            TRADE DECISION                                   │
└─────────────────────────────────────────────────────────────┘
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

# Data loaders
from src.data.binance_loader import BinanceLoader
from src.data.onchain_loader import OnChainLoader
from src.data.sentiment_loader import SentimentLoader
from src.data.coingecko_loader import CoinGeckoLoader
from src.data.funding_loader import FundingLoader
from src.data.news_loader import NewsLoader
from src.data.social_loader import SocialLoader

# Guards
from src.guards.quant_guard import QuantGuard
from src.guards.nlp_guard import NLPGuard, FusionGuard

# Cores (will be loaded when models available)
# from src.core.quant_transformer import QuantTransformer
# from src.core.nlp_finbert import NLPFinBERT
# from src.core.meta_decision import MetaDecisionEngine

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class PipelineConfig:
    """Configuration for data pipeline."""
    
    # Data settings
    lookback_days: int = 60        # For quant features
    news_lookback_hours: int = 24  # For NLP features
    
    # Update intervals
    quant_update_seconds: int = 60
    nlp_update_seconds: int = 300
    
    # Quality thresholds
    min_quant_quality: float = 0.7
    min_nlp_quality: float = 0.6
    
    # Feature settings
    use_derivatives: bool = True
    use_onchain: bool = True
    use_social: bool = True
    use_news: bool = True


# ============================================================
# DATA PIPELINE
# ============================================================

class DataPipeline:
    """
    Main data pipeline orchestrator.
    
    Collects, cleans, and feeds data to CORE models.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the data pipeline."""
        self.config = config or PipelineConfig()
        
        # Initialize loaders
        self._init_loaders()
        
        # Initialize guards
        self._init_guards()
        
        # Cache
        self._quant_cache: Optional[pd.DataFrame] = None
        self._nlp_cache: Optional[List[Dict]] = None
        self._last_quant_update: Optional[datetime] = None
        self._last_nlp_update: Optional[datetime] = None
        
        logger.info("DataPipeline initialized")
        
    def _init_loaders(self):
        """Initialize all data loaders."""
        logger.info("Initializing data loaders...")
        
        try:
            self.binance = BinanceLoader()
        except Exception as e:
            logger.warning(f"BinanceLoader init failed: {e}")
            self.binance = None
            
        try:
            self.onchain = OnChainLoader()
        except Exception as e:
            logger.warning(f"OnChainLoader init failed: {e}")
            self.onchain = None
            
        try:
            self.sentiment = SentimentLoader()
        except Exception as e:
            logger.warning(f"SentimentLoader init failed: {e}")
            self.sentiment = None
            
        try:
            self.coingecko = CoinGeckoLoader()
        except Exception as e:
            logger.warning(f"CoinGeckoLoader init failed: {e}")
            self.coingecko = None
            
        try:
            self.news = NewsLoader()
        except Exception as e:
            logger.warning(f"NewsLoader init failed: {e}")
            self.news = None
            
        try:
            self.social = SocialLoader()
        except Exception as e:
            logger.warning(f"SocialLoader init failed: {e}")
            self.social = None
            
    def _init_guards(self):
        """Initialize quality guards."""
        self.quant_guard = QuantGuard()
        self.nlp_guard = NLPGuard()
        self.fusion_guard = FusionGuard()
        
    # ========================================================
    # QUANT DATA COLLECTION
    # ========================================================
    
    def collect_quant_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Collect all quantitative data sources.
        
        Returns:
            DataFrame with all quant features
        """
        # Check cache
        if not force_refresh and self._quant_cache is not None:
            if self._last_quant_update:
                elapsed = (datetime.now() - self._last_quant_update).total_seconds()
                if elapsed < self.config.quant_update_seconds:
                    return self._quant_cache
                    
        logger.info("Collecting quant data...")
        
        features = {}
        
        # 1. Derivatives data (Binance)
        if self.config.use_derivatives and self.binance:
            try:
                deriv = self.binance.get_all_derivatives_data()
                features.update({
                    "open_interest": deriv.get("open_interest", 0),
                    "oi_change": deriv.get("oi_change", 0),
                    "long_short_ratio": deriv.get("long_short_ratio", 1.0),
                    "funding_rate": deriv.get("funding_rate", 0),
                })
            except Exception as e:
                logger.warning(f"Derivatives data fetch failed: {e}")
                
        # 2. On-chain data
        if self.config.use_onchain and self.onchain:
            try:
                netflow = self.onchain.get_latest_netflow()
                features.update({
                    "whale_netflow": netflow.get("netflow_pct", 0),
                })
            except Exception as e:
                logger.warning(f"On-chain data fetch failed: {e}")
                
        # 3. Sentiment (Fear & Greed)
        if self.sentiment:
            try:
                fg = self.sentiment.get_latest_fear_greed()
                features.update({
                    "fear_greed": fg.get("value", 50),
                })
            except Exception as e:
                logger.warning(f"Sentiment data fetch failed: {e}")
                
        # 4. Macro data (CoinGecko)
        if self.coingecko:
            try:
                global_data = self.coingecko.get_global_data()
                features.update({
                    "btc_dominance": global_data.get("btc_dominance", 50),
                    "total_market_cap": global_data.get("total_market_cap", 0),
                })
            except Exception as e:
                logger.warning(f"CoinGecko data fetch failed: {e}")
                
        # 5. Social metrics
        if self.config.use_social and self.social:
            try:
                social_features = self.social.get_social_features()
                features.update(social_features)
            except Exception as e:
                logger.warning(f"Social data fetch failed: {e}")
                
        # Create DataFrame
        df = pd.DataFrame([{
            "timestamp": datetime.now(),
            **features
        }])
        
        self._quant_cache = df
        self._last_quant_update = datetime.now()
        
        return df
    
    # ========================================================
    # NLP DATA COLLECTION
    # ========================================================
    
    def collect_nlp_data(self, force_refresh: bool = False) -> List[Dict]:
        """
        Collect all NLP/text data sources.
        
        Returns:
            List of news/text items
        """
        # Check cache
        if not force_refresh and self._nlp_cache is not None:
            if self._last_nlp_update:
                elapsed = (datetime.now() - self._last_nlp_update).total_seconds()
                if elapsed < self.config.nlp_update_seconds:
                    return self._nlp_cache
                    
        logger.info("Collecting NLP data...")
        
        all_items = []
        
        # 1. News
        if self.config.use_news and self.news:
            try:
                news_df = self.news.get_news_dataframe(hours=self.config.news_lookback_hours)
                for _, row in news_df.iterrows():
                    all_items.append({
                        "text": row.get("text", ""),
                        "source": row.get("source", "news"),
                        "timestamp": row.get("timestamp", datetime.now()),
                        "type": "news"
                    })
            except Exception as e:
                logger.warning(f"News data fetch failed: {e}")
                
        # 2. Reddit
        if self.news:
            try:
                reddit_posts = self.news.get_all_reddit_posts()
                for post in reddit_posts:
                    all_items.append({
                        "text": post.get("title", ""),
                        "source": f"reddit_{post.get('subreddit', '')}",
                        "timestamp": datetime.fromtimestamp(post.get("created_utc", 0)),
                        "type": "reddit",
                        "engagement": post.get("score", 0)
                    })
            except Exception as e:
                logger.warning(f"Reddit data fetch failed: {e}")
                
        # 3. Social/Twitter
        if self.config.use_social and self.social:
            try:
                tweets = self.social.get_all_influencer_tweets()
                for tweet in tweets:
                    all_items.append({
                        "text": tweet.get("text", ""),
                        "source": f"twitter_{tweet.get('username', '')}",
                        "timestamp": tweet.get("timestamp", ""),
                        "type": "twitter",
                        "is_influencer": tweet.get("is_influencer", False)
                    })
            except Exception as e:
                logger.warning(f"Twitter data fetch failed: {e}")
                
        self._nlp_cache = all_items
        self._last_nlp_update = datetime.now()
        
        return all_items
    
    # ========================================================
    # DATA PROCESSING
    # ========================================================
    
    def process_quant_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Process quant data through QuantGuard.
        
        Returns:
            Tuple of (cleaned_df, quality_report)
        """
        return self.quant_guard.process(df)
    
    def process_nlp_data(self, items: List[Dict]) -> Tuple[pd.DataFrame, Dict]:
        """
        Process NLP data through NLPGuard.
        
        Returns:
            Tuple of (filtered_df, quality_report)
        """
        if not items:
            return pd.DataFrame(), {"passed": 0}
            
        df = pd.DataFrame(items)
        return self.nlp_guard.process_dataframe(df, text_col="text", source_col="source")
    
    # ========================================================
    # FULL PIPELINE
    # ========================================================
    
    def run(self) -> Dict[str, Any]:
        """
        Run full data pipeline.
        
        Returns:
            Dict with processed data and quality reports
        """
        logger.info("Running full data pipeline...")
        
        # Collect data
        quant_raw = self.collect_quant_data()
        nlp_raw = self.collect_nlp_data()
        
        # Process through guards
        quant_clean, quant_report = self.process_quant_data(quant_raw)
        nlp_clean, nlp_report = self.process_nlp_data(nlp_raw)
        
        # Quality check
        quant_quality = quant_report.get("final_quality", {}).get("overall", 0)
        nlp_quality = nlp_report.get("avg_quality", 0)
        
        result = {
            "quant_data": quant_clean,
            "nlp_data": nlp_clean,
            "quant_quality": quant_quality,
            "nlp_quality": nlp_quality,
            "quant_report": quant_report,
            "nlp_report": nlp_report,
            "quant_passed": quant_quality >= self.config.min_quant_quality,
            "nlp_passed": nlp_quality >= self.config.min_nlp_quality,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(
            f"Pipeline complete: Quant quality={quant_quality:.2f}, "
            f"NLP quality={nlp_quality:.2f}, "
            f"NLP items={len(nlp_clean)}"
        )
        
        return result
    
    def get_features_for_core1(self) -> np.ndarray:
        """
        Get processed features ready for CORE 1 input.
        
        Returns:
            Feature array for QuantTransformer
        """
        quant_data = self.collect_quant_data()
        quant_clean, _ = self.process_quant_data(quant_data)
        
        # Convert to numpy array
        feature_cols = [c for c in quant_clean.columns if c != "timestamp"]
        return quant_clean[feature_cols].values
    
    def get_texts_for_core2(self) -> List[Dict]:
        """
        Get processed texts ready for CORE 2 input.
        
        Returns:
            List of text items for NLPFinBERT
        """
        nlp_raw = self.collect_nlp_data()
        nlp_clean, _ = self.process_nlp_data(nlp_raw)
        
        return nlp_clean.to_dict("records")


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_pipeline() -> DataPipeline:
    """Get initialized pipeline instance."""
    return DataPipeline()


def run_pipeline() -> Dict:
    """Run full pipeline and return results."""
    pipeline = DataPipeline()
    return pipeline.run()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Data Pipeline...")
    
    pipeline = DataPipeline()
    result = pipeline.run()
    
    print(f"\n=== Pipeline Results ===")
    print(f"Quant Quality: {result['quant_quality']:.2f}")
    print(f"NLP Quality: {result['nlp_quality']:.2f}")
    print(f"Quant Passed: {result['quant_passed']}")
    print(f"NLP Passed: {result['nlp_passed']}")
    print(f"NLP Items: {len(result['nlp_data'])}")
