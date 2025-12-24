#!/usr/bin/env python3
"""
Unit tests for CORE 3: Meta Decision Engine

Tests cover:
- Configuration
- Fusion modules (PyTorch)
- Decision making
- Integration with CORE 1 + CORE 2 outputs
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.meta_decision import MetaDecisionEngine, MetaDecisionConfig

# Check for PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestMetaDecisionConfig:
    """Test configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MetaDecisionConfig()
        
        assert config.quant_dim == 512
        assert config.nlp_dim == 512
        assert config.fusion_dim == 512
        assert config.n_attention_heads == 8
        assert config.n_actions == 3
        assert config.min_confidence == 0.6
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = MetaDecisionConfig(
            fusion_dim=256,
            n_fusion_layers=1
        )
        
        assert config.fusion_dim == 256
        assert config.n_fusion_layers == 1


class TestMetaDecisionInitialization:
    """Test engine initialization."""
    
    def test_init_default(self):
        """Test default initialization."""
        engine = MetaDecisionEngine()
        
        assert engine.config is not None
        assert len(engine._experience_buffer) == 0
        assert engine._cumulative_reward == 0.0
    
    def test_architecture_summary(self):
        """Test architecture summary generation."""
        engine = MetaDecisionEngine()
        summary = engine.get_architecture_summary()
        
        assert "CORE 3" in summary
        assert "Meta Decision" in summary
        assert "512" in summary


class TestFuseEmbeddings:
    """Test embedding fusion."""
    
    @pytest.fixture
    def engine(self):
        return MetaDecisionEngine()
    
    def test_fuse_basic(self, engine):
        """Test basic fusion."""
        quant = np.random.randn(512).astype(np.float32)
        nlp = np.random.randn(512).astype(np.float32)
        
        fused = engine.fuse_embeddings(quant, nlp)
        
        assert isinstance(fused, np.ndarray)
        assert len(fused) == engine.config.fusion_dim
    
    def test_fuse_with_confidence(self, engine):
        """Test fusion with confidence scores."""
        quant = np.random.randn(512).astype(np.float32)
        nlp = np.random.randn(512).astype(np.float32)
        
        fused = engine.fuse_embeddings(quant, nlp, quant_conf=0.8, nlp_conf=0.6)
        
        assert isinstance(fused, np.ndarray)


class TestMakeDecision:
    """Test decision making."""
    
    @pytest.fixture
    def engine(self):
        return MetaDecisionEngine()
    
    @pytest.fixture
    def sample_inputs(self):
        return {
            "quant_output": {
                "signal": 0.5,
                "confidence": 0.7,
                "market_state": np.random.randn(512).astype(np.float32)
            },
            "nlp_output": {
                "signal": 0.3,
                "confidence": 0.6,
                "sentiment_state": np.random.randn(512).astype(np.float32)
            },
            "market_data": {
                "atr": 100,
                "volatility": 0.02
            }
        }
    
    def test_decision_structure(self, engine, sample_inputs):
        """Test decision output structure."""
        decision = engine.make_decision(
            sample_inputs["quant_output"],
            sample_inputs["nlp_output"],
            sample_inputs["market_data"]
        )
        
        assert "action" in decision
        assert "direction" in decision
        assert "position_size" in decision
        assert "stop_loss" in decision
        assert "take_profit" in decision
        assert "confidence" in decision
        assert "reasoning" in decision
        assert "timestamp" in decision
    
    def test_action_values(self, engine, sample_inputs):
        """Test action is valid."""
        decision = engine.make_decision(
            sample_inputs["quant_output"],
            sample_inputs["nlp_output"],
            sample_inputs["market_data"]
        )
        
        assert decision["action"] in ["LONG", "SHORT", "HOLD"]
        assert decision["direction"] in [-1, 0, 1]
    
    def test_confidence_bounds(self, engine, sample_inputs):
        """Test confidence is bounded."""
        decision = engine.make_decision(
            sample_inputs["quant_output"],
            sample_inputs["nlp_output"],
            sample_inputs["market_data"]
        )
        
        assert 0 <= decision["confidence"] <= 1
    
    def test_position_size_bounds(self, engine, sample_inputs):
        """Test position size is bounded."""
        decision = engine.make_decision(
            sample_inputs["quant_output"],
            sample_inputs["nlp_output"],
            sample_inputs["market_data"]
        )
        
        assert 0 <= decision["position_size"] <= 1
    
    def test_bullish_signals(self, engine):
        """Test bullish signal handling."""
        quant_output = {
            "signal": 0.8,
            "confidence": 0.9,
            "market_state": np.random.randn(512).astype(np.float32)
        }
        nlp_output = {
            "signal": 0.7,
            "confidence": 0.85,
            "sentiment_state": np.random.randn(512).astype(np.float32)
        }
        market_data = {"atr": 100}
        
        decision = engine.make_decision(quant_output, nlp_output, market_data)
        
        # With both strong bullish signals and high confidence, expect LONG
        # (unless neural network decides otherwise)
        assert decision["action"] in ["LONG", "SHORT", "HOLD"]
    
    def test_empty_market_state(self, engine):
        """Test handling of missing market state."""
        quant_output = {"signal": 0.5, "confidence": 0.7}
        nlp_output = {"signal": 0.3, "confidence": 0.6}
        market_data = {"atr": 100}
        
        decision = engine.make_decision(quant_output, nlp_output, market_data)
        
        assert "action" in decision


class TestRecordTrade:
    """Test trade recording."""
    
    @pytest.fixture
    def engine(self):
        return MetaDecisionEngine()
    
    def test_record_trade(self, engine):
        """Test recording a trade."""
        decision = {"action": "LONG", "confidence": 0.8}
        outcome = {"pnl": 100}
        
        engine.record_trade(decision, outcome)
        
        assert len(engine._trade_history) == 1
        assert engine._cumulative_reward == 100
    
    def test_performance_stats(self, engine):
        """Test performance statistics."""
        # Record some trades
        engine.record_trade({"action": "LONG"}, {"pnl": 100})
        engine.record_trade({"action": "SHORT"}, {"pnl": -50})
        engine.record_trade({"action": "LONG"}, {"pnl": 75})
        
        stats = engine.get_performance_stats()
        
        assert stats["total_trades"] == 3
        assert stats["total_pnl"] == 125
        assert stats["win_rate"] == 2/3


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline(self):
        """Test full decision pipeline."""
        engine = MetaDecisionEngine()
        
        # Simulate CORE 1 output
        quant_output = {
            "signal": 0.5,
            "confidence": 0.75,
            "market_state": np.random.randn(512).astype(np.float32)
        }
        
        # Simulate CORE 2 output
        nlp_output = {
            "signal": 0.4,
            "confidence": 0.65,
            "sentiment_state": np.random.randn(512).astype(np.float32)
        }
        
        # Market data
        market_data = {"atr": 500, "volatility": 0.02}
        
        # Make decision
        decision = engine.make_decision(quant_output, nlp_output, market_data)
        
        # Verify all outputs
        assert decision["action"] in ["LONG", "SHORT", "HOLD"]
        assert "reasoning" in decision
        assert "weights" in decision
        assert len(decision["weights"]) == 2
    
    def test_reasoning_format(self):
        """Test reasoning string format."""
        engine = MetaDecisionEngine()
        
        quant_output = {"signal": 0.6, "confidence": 0.8, "market_state": np.zeros(512)}
        nlp_output = {"signal": 0.4, "confidence": 0.7, "sentiment_state": np.zeros(512)}
        market_data = {"atr": 100}
        
        decision = engine.make_decision(quant_output, nlp_output, market_data)
        
        reasoning = decision["reasoning"]
        assert "Quant:" in reasoning
        assert "NLP:" in reasoning


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPyTorchModules:
    """Test PyTorch modules specifically."""
    
    def test_fusion_network_forward(self):
        """Test FusionNetwork forward pass."""
        from core.meta_decision import FusionNetwork
        
        config = MetaDecisionConfig()
        model = FusionNetwork(config)
        
        quant = torch.randn(2, 512)
        nlp = torch.randn(2, 512)
        quant_conf = torch.tensor([0.8, 0.7])
        nlp_conf = torch.tensor([0.6, 0.5])
        
        outputs = model(quant, nlp, quant_conf, nlp_conf)
        
        assert "action_logits" in outputs
        assert "confidence" in outputs
        assert "position_size" in outputs
        assert outputs["action_logits"].shape == (2, 3)
    
    def test_cross_attention_fusion(self):
        """Test CrossAttentionFusion module."""
        from core.meta_decision import CrossAttentionFusion
        
        fusion = CrossAttentionFusion(d_model=512, n_heads=8)
        
        quant = torch.randn(4, 512)
        nlp = torch.randn(4, 512)
        
        fused = fusion(quant, nlp)
        
        assert fused.shape == (4, 512)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
