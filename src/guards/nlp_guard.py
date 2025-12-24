#!/usr/bin/env python3
"""
PhiHorizon V7.0 ML Guard 2: NLP Data Quality

Quality control for text/NLP data before feeding to CORE 2.

Features:
- Spam/bot detection
- Source credibility scoring
- Duplicate content removal
- Relevance filtering
- Language validation
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter
import hashlib

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class NLPGuardConfig:
    """Configuration for NLP Data Quality Guard."""
    
    # Spam detection
    min_text_length: int = 10
    max_text_length: int = 5000
    spam_keywords: List[str] = field(default_factory=lambda: [
        "free money", "guaranteed profit", "100x", "1000x",
        "send btc", "dm for signals", "join telegram",
        "pump group", "insider info", "free giveaway"
    ])
    
    # Source credibility
    trusted_sources: List[str] = field(default_factory=lambda: [
        "coindesk", "cointelegraph", "decrypt", "theblock",
        "bitcoin", "cryptocurrency", "cryptomarkets",
        "reuters", "bloomberg", "cnbc"
    ])
    
    low_credibility_sources: List[str] = field(default_factory=lambda: [
        "moonshot", "gem", "100x", "pump"
    ])
    
    # Relevance keywords
    crypto_keywords: List[str] = field(default_factory=lambda: [
        "bitcoin", "btc", "crypto", "cryptocurrency",
        "ethereum", "eth", "blockchain", "defi",
        "altcoin", "trading", "market", "price",
        "bull", "bear", "whale", "hodl"
    ])
    
    # Thresholds
    min_credibility_score: float = 0.5
    min_relevance_score: float = 0.3
    similarity_threshold: float = 0.85  # For deduplication
    
    # Quality
    min_quality_score: float = 0.6


# ============================================================
# NLP GUARD
# ============================================================

class NLPGuard:
    """
    ML Guard 2: Quality control for NLP/text data.
    
    Responsibilities:
    1. Filter spam and bot content
    2. Assess source credibility
    3. Remove duplicate content
    4. Ensure relevance to crypto
    5. Calculate overall quality score
    
    Used to filter/clean data before CORE 2 (NLP FinBERT).
    """
    
    def __init__(self, config: Optional[NLPGuardConfig] = None):
        """Initialize the NLP Guard."""
        self.config = config or NLPGuardConfig()
        self._seen_hashes: Set[str] = set()
        
    # ========================================================
    # SPAM DETECTION
    # ========================================================
    
    def is_spam(self, text: str) -> Tuple[bool, List[str]]:
        """
        Detect if text is spam.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (is_spam, reasons)
        """
        reasons = []
        text_lower = text.lower()
        
        # Length check
        if len(text) < self.config.min_text_length:
            reasons.append("too_short")
        if len(text) > self.config.max_text_length:
            reasons.append("too_long")
            
        # Spam keywords
        spam_found = [
            kw for kw in self.config.spam_keywords
            if kw in text_lower
        ]
        if spam_found:
            reasons.append(f"spam_keywords:{spam_found}")
            
        # Excessive caps
        if len(text) > 20:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if caps_ratio > 0.5:
                reasons.append("excessive_caps")
                
        # Excessive exclamation
        exclaim_count = text.count("!")
        if exclaim_count > 5:
            reasons.append("excessive_exclamation")
            
        # Repeated characters
        if re.search(r"(.)\1{4,}", text):
            reasons.append("repeated_chars")
            
        return len(reasons) > 0, reasons
    
    # ========================================================
    # SOURCE CREDIBILITY
    # ========================================================
    
    def calculate_credibility(self, source: str) -> float:
        """
        Calculate source credibility score.
        
        Args:
            source: Source name/identifier
            
        Returns:
            Credibility score 0-1
        """
        source_lower = source.lower()
        
        # Perfect match with trusted sources
        for trusted in self.config.trusted_sources:
            if trusted in source_lower:
                return 1.0
                
        # Check for low credibility sources
        for low_cred in self.config.low_credibility_sources:
            if low_cred in source_lower:
                return 0.2
                
        # Default medium credibility
        return 0.6
    
    # ========================================================
    # RELEVANCE SCORING
    # ========================================================
    
    def calculate_relevance(self, text: str) -> float:
        """
        Calculate relevance to crypto.
        
        Args:
            text: Input text
            
        Returns:
            Relevance score 0-1
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        if not words:
            return 0.0
            
        # Count crypto keywords
        keyword_count = sum(
            1 for kw in self.config.crypto_keywords
            if kw in text_lower
        )
        
        # Normalize by text length
        relevance = min(1.0, keyword_count / max(len(words) * 0.1, 1))
        
        return relevance
    
    # ========================================================
    # DEDUPLICATION
    # ========================================================
    
    def _text_hash(self, text: str) -> str:
        """Generate hash for text content."""
        # Normalize text
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def is_duplicate(self, text: str) -> bool:
        """
        Check if text is duplicate of previously seen content.
        
        Args:
            text: Input text
            
        Returns:
            True if duplicate
        """
        text_hash = self._text_hash(text)
        
        if text_hash in self._seen_hashes:
            return True
            
        self._seen_hashes.add(text_hash)
        return False
    
    def deduplicate_texts(self, texts: List[str]) -> List[str]:
        """Remove duplicate texts from list."""
        seen = set()
        unique = []
        
        for text in texts:
            text_hash = self._text_hash(text)
            if text_hash not in seen:
                seen.add(text_hash)
                unique.append(text)
                
        return unique
    
    # ========================================================
    # QUALITY SCORING
    # ========================================================
    
    def score_text(self, text: str, source: str = "") -> Dict:
        """
        Calculate quality score for a single text.
        
        Args:
            text: Input text
            source: Source name
            
        Returns:
            Dict with scores and pass/fail
        """
        # Spam check
        is_spam, spam_reasons = self.is_spam(text)
        spam_score = 0.0 if is_spam else 1.0
        
        # Credibility
        credibility_score = self.calculate_credibility(source)
        
        # Relevance
        relevance_score = self.calculate_relevance(text)
        
        # Duplicate
        is_dup = self.is_duplicate(text)
        dup_score = 0.0 if is_dup else 1.0
        
        # Overall (weighted)
        weights = {
            "spam": 0.3,
            "credibility": 0.25,
            "relevance": 0.25,
            "uniqueness": 0.2
        }
        
        overall = (
            spam_score * weights["spam"] +
            credibility_score * weights["credibility"] +
            relevance_score * weights["relevance"] +
            dup_score * weights["uniqueness"]
        )
        
        return {
            "spam_score": spam_score,
            "credibility_score": credibility_score,
            "relevance_score": relevance_score,
            "uniqueness_score": dup_score,
            "overall": overall,
            "is_spam": is_spam,
            "spam_reasons": spam_reasons,
            "is_duplicate": is_dup,
            "passed": overall >= self.config.min_quality_score
        }
    
    # ========================================================
    # DATAFRAME PROCESSING
    # ========================================================
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        source_col: str = "source"
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Process DataFrame of news/text data.
        
        Args:
            df: Input DataFrame
            text_col: Name of text column
            source_col: Name of source column
            
        Returns:
            Tuple of (filtered_df, quality_report)
        """
        if df.empty:
            return df, {"passed": 0, "filtered": 0, "quality": 0}
            
        # Reset duplicate tracking
        self._seen_hashes.clear()
        
        # Score each row
        scores = []
        for _, row in df.iterrows():
            text = str(row.get(text_col, ""))
            source = str(row.get(source_col, ""))
            score = self.score_text(text, source)
            scores.append(score)
            
        # Add scores to dataframe
        df_scored = df.copy()
        df_scored["nlp_quality_score"] = [s["overall"] for s in scores]
        df_scored["passed_quality"] = [s["passed"] for s in scores]
        
        # Filter
        df_filtered = df_scored[df_scored["passed_quality"]].copy()
        
        # Report
        report = {
            "total_rows": len(df),
            "passed": len(df_filtered),
            "filtered": len(df) - len(df_filtered),
            "filter_rate": (len(df) - len(df_filtered)) / len(df) if len(df) > 0 else 0,
            "avg_quality": df_scored["nlp_quality_score"].mean(),
            "spam_count": sum(1 for s in scores if s["is_spam"]),
            "duplicate_count": sum(1 for s in scores if s["is_duplicate"]),
            "low_relevance_count": sum(
                1 for s in scores
                if s["relevance_score"] < self.config.min_relevance_score
            ),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(
            f"NLP Guard: {report['passed']}/{report['total_rows']} passed "
            f"({report['filter_rate']:.1%} filtered)"
        )
        
        return df_filtered, report


# ============================================================
# FUSION GUARD
# ============================================================

class FusionGuard:
    """
    ML Guard 3: Quality control for fused data before CORE 3.
    
    Responsibilities:
    1. Confidence calibration
    2. Conflict detection (Quant vs NLP disagreement)
    3. Market regime classification
    4. Risk scoring
    """
    
    def __init__(self):
        self._conflict_history: List[bool] = []
        
    def detect_conflict(
        self,
        quant_signal: float,
        nlp_signal: float,
        threshold: float = 0.3
    ) -> Tuple[bool, str]:
        """
        Detect conflict between Quant and NLP signals.
        
        Args:
            quant_signal: Quant model output (-1 to 1, negative=bearish)
            nlp_signal: NLP model output (-1 to 1)
            threshold: Minimum difference to consider conflict
            
        Returns:
            Tuple of (is_conflict, description)
        """
        diff = abs(quant_signal - nlp_signal)
        
        # Check for opposite signals
        opposite = (quant_signal > 0 and nlp_signal < 0) or \
                   (quant_signal < 0 and nlp_signal > 0)
        
        is_conflict = opposite and diff > threshold
        
        if is_conflict:
            if quant_signal > nlp_signal:
                desc = f"Quant bullish ({quant_signal:.2f}) vs NLP bearish ({nlp_signal:.2f})"
            else:
                desc = f"Quant bearish ({quant_signal:.2f}) vs NLP bullish ({nlp_signal:.2f})"
        else:
            desc = "Aligned"
            
        self._conflict_history.append(is_conflict)
        
        return is_conflict, desc
    
    def classify_regime(
        self,
        volatility: float,
        trend_strength: float,
        correlation: float
    ) -> str:
        """
        Classify current market regime.
        
        Args:
            volatility: Current volatility (0-1 normalized)
            trend_strength: Trend strength (0-1)
            correlation: Cross-market correlation
            
        Returns:
            Regime classification
        """
        if volatility > 0.7:
            if trend_strength > 0.6:
                return "volatile_trending"
            else:
                return "volatile_ranging"
        else:
            if trend_strength > 0.6:
                return "calm_trending"
            else:
                return "calm_ranging"
    
    def calculate_risk_score(
        self,
        quant_confidence: float,
        nlp_confidence: float,
        has_conflict: bool,
        volatility: float
    ) -> float:
        """
        Calculate overall risk score.
        
        Args:
            quant_confidence: Quant model confidence (0-1)
            nlp_confidence: NLP model confidence (0-1)
            has_conflict: Whether signals conflict
            volatility: Current volatility
            
        Returns:
            Risk score 0-1 (higher = more risky)
        """
        # Low confidence = higher risk
        confidence_risk = 1 - (quant_confidence * 0.5 + nlp_confidence * 0.5)
        
        # Conflict = higher risk
        conflict_risk = 0.3 if has_conflict else 0
        
        # Volatility = higher risk
        vol_risk = volatility * 0.5
        
        total_risk = min(1.0, confidence_risk + conflict_risk + vol_risk)
        
        return total_risk
    
    def should_proceed(
        self,
        quant_output: Dict,
        nlp_output: Dict,
        market_data: Dict
    ) -> Tuple[bool, Dict]:
        """
        Determine if we should proceed with trade decision.
        
        Args:
            quant_output: Output from CORE 1
            nlp_output: Output from CORE 2
            market_data: Current market metrics
            
        Returns:
            Tuple of (should_proceed, analysis)
        """
        # Extract signals
        quant_signal = quant_output.get("signal", 0)
        quant_conf = quant_output.get("confidence", 0.5)
        nlp_signal = nlp_output.get("signal", 0)
        nlp_conf = nlp_output.get("confidence", 0.5)
        volatility = market_data.get("volatility", 0.5)
        
        # Conflict detection
        has_conflict, conflict_desc = self.detect_conflict(quant_signal, nlp_signal)
        
        # Risk score
        risk = self.calculate_risk_score(quant_conf, nlp_conf, has_conflict, volatility)
        
        # Regime
        regime = self.classify_regime(
            volatility,
            market_data.get("trend_strength", 0.5),
            market_data.get("correlation", 0.5)
        )
        
        # Decision
        proceed = risk < 0.6 and not (has_conflict and risk > 0.4)
        
        analysis = {
            "quant_signal": quant_signal,
            "nlp_signal": nlp_signal,
            "has_conflict": has_conflict,
            "conflict_desc": conflict_desc,
            "risk_score": risk,
            "regime": regime,
            "proceed": proceed,
            "reason": "Low risk, aligned signals" if proceed else "High risk or conflict"
        }
        
        return proceed, analysis


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def filter_news(df: pd.DataFrame) -> pd.DataFrame:
    """Quick filter of news DataFrame."""
    guard = NLPGuard()
    filtered, _ = guard.process_dataframe(df)
    return filtered


def check_signal_conflict(quant: float, nlp: float) -> bool:
    """Quick check for signal conflict."""
    guard = FusionGuard()
    is_conflict, _ = guard.detect_conflict(quant, nlp)
    return is_conflict


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test NLP Guard
    print("=== Testing NLP Guard ===")
    guard = NLPGuard()
    
    texts = [
        ("Bitcoin reaches new all-time high as institutional adoption grows", "coindesk"),
        ("FREE MONEY!!! SEND BTC NOW!!!", "unknown_pump_group"),
        ("ETH price analysis: bullish momentum continues", "cointelegraph"),
        ("Buy my course for 100x guaranteed profits", "telegram_spam"),
        ("Federal Reserve announces interest rate decision", "reuters"),
    ]
    
    for text, source in texts:
        score = guard.score_text(text, source)
        print(f"\n{text[:50]}...")
        print(f"  Source: {source}")
        print(f"  Quality: {score['overall']:.2f}")
        print(f"  Passed: {score['passed']}")
    
    # Test Fusion Guard
    print("\n=== Testing Fusion Guard ===")
    fusion = FusionGuard()
    
    proceed, analysis = fusion.should_proceed(
        quant_output={"signal": 0.7, "confidence": 0.8},
        nlp_output={"signal": 0.5, "confidence": 0.7},
        market_data={"volatility": 0.3, "trend_strength": 0.6, "correlation": 0.5}
    )
    
    print(f"Should proceed: {proceed}")
    print(f"Risk score: {analysis['risk_score']:.2f}")
    print(f"Regime: {analysis['regime']}")
