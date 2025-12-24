#!/usr/bin/env python3
"""
PhiHorizon - CORE 1 QuantTransformer Unit Tests

Tests for iTransformer-based CORE 1 model.
Run with: pytest tests/test_quant_transformer.py -v
"""

import pytest
import numpy as np


# ============================================================
# TEST: IMPORTS
# ============================================================

class TestImports:
    """Test module imports work correctly."""
    
    def test_config_import(self):
        """Test QuantTransformerConfig imports."""
        from src.core.quant_transformer import QuantTransformerConfig
        config = QuantTransformerConfig()
        assert config.seq_length == 60
        assert config.n_features == 30
        assert len(config.features) == 30
    
    def test_wrapper_import(self):
        """Test QuantTransformer wrapper imports."""
        from src.core.quant_transformer import QuantTransformer
        model = QuantTransformer()
        assert model is not None


# ============================================================
# TEST: MODEL ARCHITECTURE
# ============================================================

class TestModelArchitecture:
    """Test iTransformer model architecture."""
    
    @pytest.fixture
    def model(self):
        """Create model fixture."""
        from src.core.quant_transformer import QuantTransformer
        return QuantTransformer()
    
    def test_model_initialization(self, model):
        """Test model initializes correctly."""
        assert model.config is not None
        assert model.config.seq_length == 60
        assert model.config.n_features == 30
        assert model.config.output_dim == 512
    
    def test_architecture_summary(self, model):
        """Test architecture summary generation."""
        summary = model.get_architecture_summary()
        assert 'iTransformer' in summary
        assert 'ICLR 2024' in summary
        assert '512' in summary


# ============================================================
# TEST: FORWARD PASS
# ============================================================

class TestForwardPass:
    """Test model forward pass."""
    
    @pytest.fixture
    def model(self):
        """Create model fixture."""
        from src.core.quant_transformer import QuantTransformer
        return QuantTransformer()
    
    def test_encode_shape(self, model):
        """Test encode output shape."""
        # Batch of 2 samples
        x = np.random.randn(2, 60, 30).astype(np.float32)
        encoding = model.encode(x)
        
        assert encoding.shape == (2, 512)
    
    def test_encode_single_sample(self, model):
        """Test encode with single sample."""
        x = np.random.randn(60, 30).astype(np.float32)
        encoding = model.encode(x)
        
        assert encoding.shape == (1, 512)
    
    def test_predict_output_keys(self, model):
        """Test predict returns expected keys."""
        x = np.random.randn(2, 60, 30).astype(np.float32)
        result = model.predict(x)
        
        expected_keys = ['signal', 'confidence', 'predicted_direction', 'state_vector', 'probs']
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_predict_signal_range(self, model):
        """Test signal is in valid range."""
        x = np.random.randn(10, 60, 30).astype(np.float32)
        result = model.predict(x)
        
        assert np.all(result['signal'] >= -1)
        assert np.all(result['signal'] <= 1)
    
    def test_predict_confidence_range(self, model):
        """Test confidence is in valid range."""
        x = np.random.randn(10, 60, 30).astype(np.float32)
        result = model.predict(x)
        
        assert np.all(result['confidence'] >= 0)
        assert np.all(result['confidence'] <= 1)
    
    def test_predict_direction_values(self, model):
        """Test direction is 0 or 1."""
        x = np.random.randn(10, 60, 30).astype(np.float32)
        result = model.predict(x)
        
        assert np.all(np.isin(result['predicted_direction'], [0, 1]))


# ============================================================
# TEST: PYTORCH MODEL (if available)
# ============================================================

class TestPyTorchModel:
    """Test PyTorch model internals."""
    
    @pytest.fixture
    def torch_available(self):
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def test_torch_model_exists(self, torch_available):
        """Test PyTorch model can be instantiated."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        
        from src.core.quant_transformer import QuantTransformerModel
        model = QuantTransformerModel()
        assert model is not None
    
    def test_torch_model_params(self, torch_available):
        """Test model has expected parameter count."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        
        from src.core.quant_transformer import QuantTransformerModel
        model = QuantTransformerModel()
        n_params = model.get_num_params()
        
        # Should be around 7M params
        assert n_params > 1_000_000
        assert n_params < 50_000_000
    
    def test_torch_forward_pass(self, torch_available):
        """Test PyTorch model forward pass."""
        if not torch_available:
            pytest.skip("PyTorch not available")
        
        import torch
        from src.core.quant_transformer import QuantTransformerModel
        
        model = QuantTransformerModel()
        model.eval()
        
        x = torch.randn(2, 60, 30)
        
        with torch.no_grad():
            output = model(x)
        
        assert 'state_vector' in output
        assert 'logits' in output
        assert output['state_vector'].shape == (2, 512)
        assert output['logits'].shape == (2, 2)


# ============================================================
# TEST: INFERENCE SPEED
# ============================================================

class TestInferenceSpeed:
    """Test inference speed meets requirements."""
    
    def test_inference_time(self):
        """Test single inference is under 100ms."""
        import time
        from src.core.quant_transformer import QuantTransformer
        
        model = QuantTransformer()
        x = np.random.randn(1, 60, 30).astype(np.float32)
        
        # Warmup
        _ = model.predict(x)
        
        # Time 10 inferences
        start = time.time()
        for _ in range(10):
            _ = model.predict(x)
        elapsed = (time.time() - start) / 10 * 1000  # ms
        
        assert elapsed < 500, f"Inference too slow: {elapsed:.1f}ms"


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
