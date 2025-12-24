#!/usr/bin/env python3
"""
Unit tests for CORE 2: NLP FinBERT

Tests cover:
- Configuration
- Model initialization
- Sentiment prediction (placeholder mode)
- News encoding
- Batch processing
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.nlp_finbert import NLPFinBERT, NLPFinBERTConfig


class TestNLPFinBERTConfig:
    """Test configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = NLPFinBERTConfig()
        
        assert config.base_model == "burakutf/finetuned-finbert-crypto"
        assert config.hidden_size == 768
        assert config.output_dim == 512
        assert config.max_length == 512
        assert config.max_news_items == 50
        assert len(config.labels) == 3
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = NLPFinBERTConfig(
            max_length=256,
            output_dim=256,
            batch_size=4
        )
        
        assert config.max_length == 256
        assert config.output_dim == 256
        assert config.batch_size == 4


class TestNLPFinBERTInitialization:
    """Test model initialization."""
    
    def test_init_default(self):
        """Test default initialization."""
        model = NLPFinBERT()
        
        assert model.config is not None
        assert model.is_loaded == False
        assert model.model is None
        assert model.tokenizer is None
    
    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = NLPFinBERTConfig(output_dim=256)
        model = NLPFinBERT(config=config)
        
        assert model.config.output_dim == 256
    
    def test_architecture_summary(self):
        """Test architecture summary generation."""
        model = NLPFinBERT()
        summary = model.get_architecture_summary()
        
        assert "CORE 2" in summary
        assert "NLP FinBERT" in summary
        assert "512" in summary  # output_dim


class TestNLPFinBERTPlaceholder:
    """Test placeholder mode (without loading real model)."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return NLPFinBERT()
    
    def test_predict_sentiment_placeholder(self, model):
        """Test sentiment prediction in placeholder mode."""
        result = model.predict_sentiment("Bitcoin is doing great!")
        
        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["label"] in ["positive", "negative", "neutral"]
        assert 0 <= result["confidence"] <= 1
    
    def test_encode_text_placeholder(self, model):
        """Test text encoding in placeholder mode."""
        encoding = model.encode_text("Test text")
        
        assert isinstance(encoding, np.ndarray)
        assert encoding.shape == (model.config.hidden_size,)
    
    def test_encode_news_empty(self, model):
        """Test encoding empty news list."""
        encoding = model.encode_news([])
        
        assert isinstance(encoding, np.ndarray)
        assert encoding.shape == (model.config.output_dim,)
        assert np.allclose(encoding, 0)
    
    def test_encode_news_placeholder(self, model):
        """Test news encoding in placeholder mode."""
        news = [
            {"text": "Bitcoin surges to new high"},
            {"text": "Crypto market looks bullish"}
        ]
        encoding = model.encode_news(news)
        
        assert isinstance(encoding, np.ndarray)
        assert encoding.shape == (model.config.output_dim,)
    
    def test_predict_placeholder(self, model):
        """Test full prediction in placeholder mode."""
        news = [
            {"text": "Great news for crypto!"},
            {"text": "Market is crashing"},
            {"text": "Neutral developments"}
        ]
        result = model.predict(news)
        
        assert "signal" in result
        assert "confidence" in result
        assert "sentiment_state" in result
        assert "dominant_sentiment" in result
        assert "news_count" in result
        
        assert -1 <= result["signal"] <= 1
        assert 0 <= result["confidence"] <= 1
        assert result["news_count"] == 3
    
    def test_predict_empty_news(self, model):
        """Test prediction with empty news."""
        result = model.predict([])
        
        assert result["signal"] == 0.0
        assert result["confidence"] == 0.5
        assert result["dominant_sentiment"] == "neutral"
        assert result["news_count"] == 0


class TestNLPFinBERTEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return NLPFinBERT()
    
    def test_empty_text(self, model):
        """Test with empty text."""
        result = model.predict_sentiment("")
        assert "label" in result
    
    def test_very_long_text(self, model):
        """Test with very long text (should be truncated)."""
        long_text = "Bitcoin " * 1000  # Very long
        result = model.predict_sentiment(long_text)
        assert "label" in result
    
    def test_special_characters(self, model):
        """Test with special characters."""
        text = "🚀 Bitcoin to the moon! @BTC #crypto $$$"
        result = model.predict_sentiment(text)
        assert "label" in result
    
    def test_news_with_missing_text(self, model):
        """Test news items with missing text field."""
        news = [
            {"source": "test"},  # No text
            {"text": "Valid news"},
            {"text": ""},  # Empty text
        ]
        result = model.predict(news)
        assert "signal" in result
    
    def test_max_news_items(self, model):
        """Test respecting max_news_items limit."""
        # Create more than max items
        news = [{"text": f"News {i}"} for i in range(100)]
        encoding = model.encode_news(news)
        
        # Should still work, just use max items
        assert encoding.shape == (model.config.output_dim,)


class TestNLPFinBERTOutput:
    """Test output format and values."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return NLPFinBERT()
    
    def test_sentiment_probabilities_sum_to_one(self, model):
        """Test that sentiment probabilities sum to ~1."""
        result = model.predict_sentiment("Test text")
        probs = result["probabilities"]
        
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01
    
    def test_sentiment_state_dimensions(self, model):
        """Test sentiment state vector dimensions."""
        news = [{"text": "Test news"}]
        result = model.predict(news)
        
        state = result["sentiment_state"]
        assert state.shape == (model.config.output_dim,)
    
    def test_signal_bounds(self, model):
        """Test signal is bounded between -1 and 1."""
        news = [{"text": f"News {i}"} for i in range(10)]
        result = model.predict(news)
        
        assert -1 <= result["signal"] <= 1


class TestNLPFinBERTIntegration:
    """Integration tests."""
    
    def test_full_pipeline(self):
        """Test full inference pipeline."""
        model = NLPFinBERT()
        
        # Simulate news batch
        news = [
            {"text": "Bitcoin breaks $100K milestone", "source": "coindesk", "timestamp": "2024-01-01"},
            {"text": "SEC approves spot Bitcoin ETF", "source": "reuters", "timestamp": "2024-01-02"},
            {"text": "Market shows signs of correction", "source": "cointelegraph", "timestamp": "2024-01-03"},
        ]
        
        # Run prediction
        result = model.predict(news)
        
        # Verify all expected fields
        assert "signal" in result
        assert "confidence" in result
        assert "sentiment_state" in result
        assert "dominant_sentiment" in result
        assert "news_count" in result
        
        # Verify sentiment state can be used (e.g., passed to CORE 3)
        sentiment_state = result["sentiment_state"]
        assert isinstance(sentiment_state, np.ndarray)
        assert sentiment_state.dtype in [np.float32, np.float64]
    
    def test_reproducibility(self):
        """Test that same input gives same structure output."""
        model = NLPFinBERT()
        
        news = [{"text": "Test news article"}]
        
        result1 = model.predict(news)
        result2 = model.predict(news)
        
        # Structure should be same
        assert result1.keys() == result2.keys()
        assert result1["news_count"] == result2["news_count"]


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
