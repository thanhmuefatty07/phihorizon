#!/usr/bin/env python3
"""
PhiHorizon V7.0 CORE 2: NLP FinBERT

Production-ready sentiment analysis model for crypto news.

Architecture:
- Base: burakutf/finetuned-finbert-crypto (HuggingFace)
- Attention-based news aggregation
- Output: 512-dim Sentiment State Vector

Dependencies:
- transformers>=4.30.0
- torch>=2.0.0
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Check for transformers availability
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Install with: pip install transformers torch")


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class NLPFinBERTConfig:
    """Configuration for NLP FinBERT (CORE 2)."""
    
    # Model settings - use crypto-specific FinBERT
    base_model: str = "burakutf/finetuned-finbert-crypto"
    fallback_model: str = "ProsusAI/finbert"
    
    # Local cache path (optional - to avoid re-downloading)
    local_cache_path: Optional[str] = None
    
    max_length: int = 512          # Max tokens per text
    max_news_items: int = 50       # Max news items to process
    
    # Architecture
    hidden_size: int = 768         # FinBERT hidden size
    output_dim: int = 512          # Sentiment state vector
    n_attention_heads: int = 8     # For news aggregation
    dropout: float = 0.1
    
    # Inference
    batch_size: int = 8
    device: str = "cuda" if TRANSFORMERS_AVAILABLE and torch.cuda.is_available() else "cpu"
    
    # Sentiment labels - UPDATED to match actual model order
    # burakutf/finetuned-finbert-crypto: 0=Neutral, 1=Positive, 2=Negative
    labels: List[str] = field(default_factory=lambda: [
        "neutral", "positive", "negative"
    ])
    
    # Label mapping - synced with actual model id2label
    # Will be updated dynamically when model loads
    label_map: Dict[int, str] = field(default_factory=lambda: {
        0: "neutral",
        1: "positive", 
        2: "negative"
    })


# ============================================================
# ATTENTION AGGREGATION MODULE
# ============================================================

if TRANSFORMERS_AVAILABLE:
    class AttentionAggregation(nn.Module):
        """Attention-weighted aggregation of multiple news embeddings."""
        
        def __init__(self, hidden_size: int = 768, n_heads: int = 8, dropout: float = 0.1):
            super().__init__()
            self.hidden_size = hidden_size
            
            # Multi-head self-attention
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
            
            # Layer norm
            self.norm = nn.LayerNorm(hidden_size)
            
            # Learnable query for aggregation
            self.agg_query = nn.Parameter(torch.randn(1, 1, hidden_size))
            
        def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Aggregate multiple embeddings into single vector.
            
            Args:
                embeddings: (batch, n_news, hidden_size)
                mask: (batch, n_news) - True for padding
                
            Returns:
                (batch, hidden_size) - Aggregated embedding
            """
            batch_size = embeddings.size(0)
            
            # Expand aggregation query
            query = self.agg_query.expand(batch_size, -1, -1)
            
            # Attention: query attends to all news embeddings
            attn_output, _ = self.attention(query, embeddings, embeddings, key_padding_mask=mask)
            
            # Residual + norm
            output = self.norm(attn_output.squeeze(1))
            
            return output


# ============================================================
# NLP FINBERT MODEL
# ============================================================

class NLPFinBERT:
    """
    CORE 2: NLP FinBERT for crypto sentiment analysis.
    
    Uses pre-trained burakutf/finetuned-finbert-crypto model
    for crypto-specific sentiment classification.
    
    Output: 512-dim sentiment state vector for CORE 3 fusion.
    """
    
    def __init__(self, config: Optional[NLPFinBERTConfig] = None):
        """Initialize the NLP FinBERT model."""
        self.config = config or NLPFinBERTConfig()
        self.is_loaded = False
        self.model = None
        self.tokenizer = None
        self.aggregator = None
        self.projection = None
        
        logger.info(
            f"NLPFinBERT initialized: model={self.config.base_model}, "
            f"device={self.config.device}"
        )
    
    def load_model(self, model_name: Optional[str] = None, use_cache: bool = True) -> bool:
        """
        Load pre-trained model from HuggingFace or local cache.
        
        Args:
            model_name: Override model name (optional)
            use_cache: Try local cache first if available
            
        Returns:
            True if loaded successfully
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers not available. Cannot load model.")
            return False
        
        model_name = model_name or self.config.base_model
        
        try:
            # Try local cache first
            if use_cache and self.config.local_cache_path:
                cache_path = self.config.local_cache_path
                if os.path.exists(cache_path):
                    logger.info(f"Loading from local cache: {cache_path}")
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(cache_path)
                        self.model = AutoModelForSequenceClassification.from_pretrained(cache_path)
                        logger.info("Loaded from local cache successfully!")
                    except Exception as e:
                        logger.warning(f"Local cache failed: {e}. Falling back to HuggingFace...")
                        self.tokenizer = None
                        self.model = None
            
            # Load from HuggingFace if not cached
            if self.model is None:
                logger.info(f"Loading model from HuggingFace: {model_name}")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                except Exception as e:
                    logger.warning(f"Primary model failed: {e}. Trying fallback...")
                    model_name = self.config.fallback_model
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move to device
            self.model = self.model.to(self.config.device)
            self.model.eval()
            
            # DYNAMIC LABEL SYNC from model config
            if hasattr(self.model.config, 'id2label') and self.model.config.id2label:
                self.config.label_map = {
                    int(k): v.lower() for k, v in self.model.config.id2label.items()
                }
                self.config.labels = [
                    self.config.label_map.get(i, f"label_{i}") 
                    for i in range(len(self.config.label_map))
                ]
                logger.info(f"Labels synced from model: {self.config.labels}")
            
            # Initialize aggregation layer
            self.aggregator = AttentionAggregation(
                hidden_size=self.config.hidden_size,
                n_heads=self.config.n_attention_heads,
                dropout=self.config.dropout
            ).to(self.config.device)
            
            # Projection to output dimension
            self.projection = nn.Linear(
                self.config.hidden_size, 
                self.config.output_dim
            ).to(self.config.device)
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def save_model_cache(self, cache_path: str) -> bool:
        """
        Save model to local cache for faster future loading.
        
        Args:
            cache_path: Directory to save model
            
        Returns:
            True if saved successfully
        """
        if not self.is_loaded:
            logger.error("Cannot save - model not loaded")
            return False
        
        try:
            import os
            os.makedirs(cache_path, exist_ok=True)
            
            self.model.save_pretrained(cache_path)
            self.tokenizer.save_pretrained(cache_path)
            
            # Update config
            self.config.local_cache_path = cache_path
            
            logger.info(f"Model saved to: {cache_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get embeddings for list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Tensor of shape (n_texts, hidden_size)
        """
        if not self.is_loaded:
            return torch.randn(len(texts), self.config.hidden_size)
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            ).to(self.config.device)
            
            # Get embeddings (CLS token)
            with torch.no_grad():
                outputs = self.model.base_model(**inputs)
                # Get CLS token embedding
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embeddings)
        
        return torch.cat(embeddings, dim=0)
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode single text to vector.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector (hidden_size,)
        """
        if not self.is_loaded:
            logger.warning("Model not loaded, returning random encoding")
            return np.random.randn(self.config.hidden_size).astype(np.float32)
        
        embeddings = self._get_embeddings([text])
        return embeddings[0].cpu().numpy()
    
    def encode_news(self, news_items: List[Dict]) -> np.ndarray:
        """
        Encode multiple news items with attention aggregation.
        
        Args:
            news_items: List of {"text": str, "source": str, ...}
            
        Returns:
            Aggregated sentiment state vector (output_dim,)
        """
        if not news_items:
            return np.zeros(self.config.output_dim, dtype=np.float32)
        
        if not self.is_loaded:
            logger.warning("Model not loaded, returning random encoding")
            return np.random.randn(self.config.output_dim).astype(np.float32)
        
        # Extract texts
        texts = [item.get("text", "") for item in news_items[:self.config.max_news_items]]
        texts = [t for t in texts if t.strip()]  # Remove empty
        
        if not texts:
            return np.zeros(self.config.output_dim, dtype=np.float32)
        
        # Get embeddings
        embeddings = self._get_embeddings(texts)  # (n_news, hidden_size)
        
        # Add batch dimension and aggregate
        embeddings = embeddings.unsqueeze(0)  # (1, n_news, hidden_size)
        
        with torch.no_grad():
            aggregated = self.aggregator(embeddings)  # (1, hidden_size)
            projected = self.projection(aggregated)   # (1, output_dim)
        
        return projected.squeeze(0).cpu().numpy()
    
    def predict_sentiment(self, text: str) -> Dict:
        """
        Predict sentiment for single text.
        
        Args:
            text: Input text
            
        Returns:
            Dict with label, confidence, probabilities
        """
        if not self.is_loaded:
            # Placeholder prediction
            probs = np.random.dirichlet([1, 1, 1])
            label_idx = np.argmax(probs)
            return {
                "label": self.config.labels[label_idx],
                "confidence": float(probs[label_idx]),
                "probabilities": {
                    label: float(p) 
                    for label, p in zip(self.config.labels, probs)
                }
            }
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        ).to(self.config.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).squeeze(0).cpu().numpy()
        
        # Get label
        label_idx = int(np.argmax(probs))
        label = self.config.label_map.get(label_idx, self.config.labels[label_idx])
        
        return {
            "label": label,
            "confidence": float(probs[label_idx]),
            "probabilities": {
                self.config.label_map.get(i, self.config.labels[i]): float(p)
                for i, p in enumerate(probs)
            }
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict sentiment for batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction dicts
        """
        if not self.is_loaded or not texts:
            return [self.predict_sentiment(t) for t in texts]
        
        all_results = []
        
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            ).to(self.config.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=1).cpu().numpy()
            
            for j, text_probs in enumerate(probs):
                label_idx = int(np.argmax(text_probs))
                label = self.config.label_map.get(label_idx, self.config.labels[label_idx])
                
                all_results.append({
                    "label": label,
                    "confidence": float(text_probs[label_idx]),
                    "probabilities": {
                        self.config.label_map.get(k, self.config.labels[k]): float(p)
                        for k, p in enumerate(text_probs)
                    }
                })
        
        return all_results
    
    def predict(self, news_items: List[Dict]) -> Dict:
        """
        Make prediction from news items.
        
        Args:
            news_items: List of news dictionaries
            
        Returns:
            Dict with signal, confidence, sentiment_state
        """
        # Get aggregated encoding
        encoding = self.encode_news(news_items)
        
        # Get individual sentiments
        texts = [item.get("text", "") for item in news_items if item.get("text")]
        
        if not texts:
            return {
                "signal": 0.0,
                "confidence": 0.5,
                "sentiment_state": encoding,
                "dominant_sentiment": "neutral",
                "news_count": 0
            }
        
        # Batch predict sentiments
        sentiments = self.predict_batch(texts)
        
        # Calculate overall sentiment signal
        pos_score = sum(s["probabilities"].get("positive", 0) for s in sentiments)
        neg_score = sum(s["probabilities"].get("negative", 0) for s in sentiments)
        total = len(sentiments)
        
        # Signal: -1 (bearish) to +1 (bullish)
        signal = (pos_score - neg_score) / total if total > 0 else 0
        
        # Average confidence
        avg_confidence = np.mean([s["confidence"] for s in sentiments])
        
        # Dominant sentiment
        pos_count = sum(1 for s in sentiments if s["label"] == "positive")
        neg_count = sum(1 for s in sentiments if s["label"] == "negative")
        
        if pos_count > neg_count:
            dominant = "positive"
        elif neg_count > pos_count:
            dominant = "negative"
        else:
            dominant = "neutral"
        
        return {
            "signal": float(np.clip(signal, -1, 1)),
            "confidence": float(avg_confidence),
            "sentiment_state": encoding,
            "dominant_sentiment": dominant,
            "news_count": len(news_items),
            "sentiment_breakdown": {
                "positive": pos_count,
                "negative": neg_count,
                "neutral": total - pos_count - neg_count
            }
        }
    
    def get_architecture_summary(self) -> str:
        """Get model architecture summary."""
        return f"""
CORE 2: NLP FinBERT
===================
Base Model: {self.config.base_model}
Device: {self.config.device}
Hidden Size: {self.config.hidden_size}
Max Length: {self.config.max_length} tokens
Max News: {self.config.max_news_items} items
Output: {self.config.output_dim}-dim Sentiment State Vector

Aggregation: Multi-head Attention ({self.config.n_attention_heads} heads)
Labels: {', '.join(self.config.labels)}
Status: {'Loaded' if self.is_loaded else 'Not loaded'}
"""


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("CORE 2: NLP FinBERT Test")
    print("=" * 60)
    
    model = NLPFinBERT()
    print(model.get_architecture_summary())
    
    # Try to load model
    print("\nLoading model...")
    success = model.load_model()
    
    if success:
        # Test with real inference
        test_texts = [
            "Bitcoin breaks $100K! Massive bullish momentum!",
            "Crypto market crashes as SEC announces crackdown",
            "Ethereum developers announce new upgrade timeline"
        ]
        
        print("\nSingle predictions:")
        for text in test_texts:
            result = model.predict_sentiment(text)
            print(f"  '{text[:50]}...'")
            print(f"    → {result['label']} ({result['confidence']:.2%})")
        
        # Test batch
        print("\nBatch prediction:")
        news_items = [{"text": t, "source": "test"} for t in test_texts]
        result = model.predict(news_items)
        print(f"  Signal: {result['signal']:.4f}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Dominant: {result['dominant_sentiment']}")
        print(f"  Breakdown: {result['sentiment_breakdown']}")
        print(f"  State shape: {result['sentiment_state'].shape}")
    else:
        print("\nModel not loaded, testing with placeholders...")
        dummy_news = [
            {"text": "Bitcoin reaches new all-time high", "source": "coindesk"},
            {"text": "Crypto market shows bullish momentum", "source": "cointelegraph"},
        ]
        
        result = model.predict(dummy_news)
        print(f"  Signal: {result['signal']:.4f}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Dominant: {result['dominant_sentiment']}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
