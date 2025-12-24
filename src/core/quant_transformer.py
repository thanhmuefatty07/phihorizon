#!/usr/bin/env python3
"""
PhiHorizon V7.0 CORE 1: Quant Transformer

iTransformer-based model for processing numerical market data.
Architecture based on ICLR 2024 paper: "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"

Key Innovation:
- Traditional Transformer: Attention over TIME tokens
- iTransformer: Attention over VARIATE tokens (features)
- Better captures multivariate correlations in financial data

To be trained on Kaggle with GPU.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

# Try importing PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Model will use placeholder mode.")


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class QuantTransformerConfig:
    """Configuration for Quant Transformer (CORE 1) - iTransformer Architecture."""
    
    # Sequence settings
    seq_length: int = 60           # 60-day lookback
    n_features: int = 30           # Number of input features (variates)
    
    # iTransformer Architecture
    d_model: int = 256             # Embedding dimension per variate
    n_heads: int = 8               # Attention heads
    n_layers: int = 6              # Transformer layers (reduced for efficiency)
    d_ff: int = 512                # Feedforward dimension
    dropout: float = 0.1           # Dropout rate
    
    # Output
    output_dim: int = 512          # Market state vector dimension
    n_classes: int = 2             # Binary classification (UP/DOWN)
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    max_epochs: int = 100
    early_stop_patience: int = 10
    weight_decay: float = 1e-5
    
    # Feature list (30 features for ETH-USDT)
    features: List[str] = field(default_factory=lambda: [
        # Price features (10)
        "open", "high", "low", "close", "volume",
        "return_1d", "return_7d", "return_30d",
        "volatility_7d", "volatility_30d",
        
        # Technical indicators (9)
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_lower", "bb_width",
        "atr_14", "adx_14",
        
        # Derivatives (5)
        "funding_rate", "open_interest", "oi_change",
        "long_short_ratio", "top_trader_ratio",
        
        # On-chain (4)
        "hash_rate", "hash_rate_change",
        "whale_netflow", "exchange_reserve",
        
        # Macro (2)
        "btc_dominance", "fear_greed"
    ])


# ============================================================
# ITRANSFORMER PYTORCH MODEL
# ============================================================

if TORCH_AVAILABLE:
    
    class VariateEmbedding(nn.Module):
        """
        Embed each variate (feature) independently into d_model dimensions.
        
        Unlike traditional transformers that embed time tokens,
        iTransformer embeds each variate's entire time series.
        """
        
        def __init__(self, seq_length: int, d_model: int, dropout: float = 0.1):
            super().__init__()
            self.seq_length = seq_length
            self.d_model = d_model
            
            # Linear projection from seq_length to d_model
            self.projection = nn.Linear(seq_length, d_model)
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(d_model)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch, seq_length, n_features)
                
            Returns:
                (batch, n_features, d_model) - each feature becomes a token
            """
            # Transpose to (batch, n_features, seq_length)
            x = x.transpose(1, 2)
            
            # Project each variate: (batch, n_features, d_model)
            x = self.projection(x)
            x = self.dropout(x)
            x = self.layer_norm(x)
            
            return x
    
    
    class MultiHeadAttention(nn.Module):
        """Multi-head self-attention for variate tokens."""
        
        def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
            super().__init__()
            assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
            
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
            
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Self-attention over variate tokens.
            
            Args:
                x: (batch, n_features, d_model)
                
            Returns:
                (batch, n_features, d_model)
            """
            batch_size, n_tokens, _ = x.shape
            
            # Linear projections
            Q = self.W_q(x).view(batch_size, n_tokens, self.n_heads, self.d_k).transpose(1, 2)
            K = self.W_k(x).view(batch_size, n_tokens, self.n_heads, self.d_k).transpose(1, 2)
            V = self.W_v(x).view(batch_size, n_tokens, self.n_heads, self.d_k).transpose(1, 2)
            
            # Scaled dot-product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            context = torch.matmul(attn_weights, V)
            
            # Concatenate heads
            context = context.transpose(1, 2).contiguous().view(batch_size, n_tokens, self.d_model)
            
            # Final projection
            output = self.W_o(context)
            
            return output
    
    
    class FeedForward(nn.Module):
        """Position-wise feed-forward network."""
        
        def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
            super().__init__()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)
            self.activation = nn.GELU()
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.linear1(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear2(x)
            return x
    
    
    class ITransformerBlock(nn.Module):
        """Single iTransformer encoder block."""
        
        def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
            super().__init__()
            
            self.attention = MultiHeadAttention(d_model, n_heads, dropout)
            self.feed_forward = FeedForward(d_model, d_ff, dropout)
            
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Self-attention with residual
            attn_out = self.attention(x)
            x = self.norm1(x + self.dropout(attn_out))
            
            # Feed-forward with residual
            ff_out = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_out))
            
            return x
    
    
    class QuantTransformerModel(nn.Module):
        """
        CORE 1: iTransformer for Market State Encoding
        
        Architecture (iTransformer - ICLR 2024):
        1. Variate Embedding: Each feature → d_model dim token
        2. iTransformer Blocks: Attention ACROSS variates (not time)
        3. Global Pooling: Aggregate all variate representations
        4. Output Head: Market state vector + classification
        
        Key Insight:
        - Traditional Transformer struggles with time series because
          attending over time lacks meaning (time is ordinal, not semantic)
        - iTransformer inverts this: attend over variates (features)
        - Each variate's time pattern is learned via FFN, not attention
        """
        
        def __init__(self, config: Optional[QuantTransformerConfig] = None):
            super().__init__()
            self.config = config or QuantTransformerConfig()
            
            # Variate embedding
            self.variate_embed = VariateEmbedding(
                seq_length=self.config.seq_length,
                d_model=self.config.d_model,
                dropout=self.config.dropout
            )
            
            # iTransformer encoder blocks
            self.encoder_blocks = nn.ModuleList([
                ITransformerBlock(
                    d_model=self.config.d_model,
                    n_heads=self.config.n_heads,
                    d_ff=self.config.d_ff,
                    dropout=self.config.dropout
                )
                for _ in range(self.config.n_layers)
            ])
            
            # Output projection
            self.output_proj = nn.Sequential(
                nn.Linear(self.config.d_model * self.config.n_features, self.config.d_ff),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.d_ff, self.config.output_dim)
            )
            
            # Classification head (for training)
            self.classifier = nn.Linear(self.config.output_dim, self.config.n_classes)
            
            # Initialize weights
            self._init_weights()
            
        def _init_weights(self):
            """Initialize weights with Xavier/Glorot."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
        
        def encode(self, x: torch.Tensor) -> torch.Tensor:
            """
            Encode market features into state vector.
            
            Args:
                x: Input features (batch, seq_length, n_features)
                
            Returns:
                Market state vector (batch, output_dim)
            """
            # Embed variates: (batch, n_features, d_model)
            x = self.variate_embed(x)
            
            # Apply iTransformer blocks
            for block in self.encoder_blocks:
                x = block(x)
            
            # Flatten: (batch, n_features * d_model)
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)
            
            # Project to output dimension: (batch, output_dim)
            state_vector = self.output_proj(x)
            
            return state_vector
        
        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            Full forward pass with classification.
            
            Args:
                x: Input features (batch, seq_length, n_features)
                
            Returns:
                Dict with state_vector and logits
            """
            state_vector = self.encode(x)
            logits = self.classifier(state_vector)
            
            return {
                'state_vector': state_vector,
                'logits': logits,
                'probs': F.softmax(logits, dim=-1)
            }
        
        def predict(self, x: torch.Tensor) -> Dict:
            """Inference method for production."""
            self.eval()
            with torch.no_grad():
                output = self.forward(x)
                probs = output['probs']
                
                # Direction: 0=DOWN, 1=UP
                predicted_class = torch.argmax(probs, dim=-1)
                confidence = torch.max(probs, dim=-1).values
                
                # Signal: -1 to 1
                signal = probs[:, 1] - probs[:, 0]
                
            return {
                'signal': signal.cpu().numpy(),
                'confidence': confidence.cpu().numpy(),
                'predicted_direction': predicted_class.cpu().numpy(),
                'state_vector': output['state_vector'].cpu().numpy(),
                'probs': probs.cpu().numpy()
            }
        
        def get_num_params(self) -> int:
            """Count trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# WRAPPER CLASS (Backward Compatible)
# ============================================================

class QuantTransformer:
    """
    CORE 1: Quant Transformer wrapper for inference.
    
    This class wraps the PyTorch model and handles:
    - Model loading from checkpoint
    - Numpy to Tensor conversion
    - Device management (CPU/GPU)
    """
    
    def __init__(self, config: Optional[QuantTransformerConfig] = None):
        """Initialize the Quant Transformer."""
        self.config = config or QuantTransformerConfig()
        self.is_loaded = False
        self.model = None
        self.device = 'cpu'
        
        if TORCH_AVAILABLE:
            # Check for GPU
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            
            # Initialize model
            self.model = QuantTransformerModel(self.config)
            self.model.to(self.device)
            
            logger.info(
                f"QuantTransformer (iTransformer) initialized: "
                f"{self.config.n_features} features, "
                f"{self.config.seq_length} days lookback, "
                f"{self.config.n_layers} layers, "
                f"device={self.device}, "
                f"params={self.model.get_num_params():,}"
            )
        else:
            logger.warning("PyTorch not available. Running in placeholder mode.")
            
    def load_model(self, path: str) -> bool:
        """
        Load trained model from checkpoint.
        
        Args:
            path: Path to .pt or .pth checkpoint file
            
        Returns:
            True if loaded successfully
        """
        if not TORCH_AVAILABLE:
            logger.error("Cannot load model: PyTorch not available")
            return False
            
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            self.is_loaded = True
            logger.info(f"Model loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
            
    def encode(self, features: np.ndarray) -> np.ndarray:
        """
        Encode market features into state vector.
        
        Args:
            features: Shape (batch, seq_length, n_features) or (seq_length, n_features)
            
        Returns:
            Market state vectors, shape (batch, output_dim)
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, returning random encoding")
            batch_size = features.shape[0] if features.ndim == 3 else 1
            return np.random.randn(batch_size, self.config.output_dim)
        
        # Ensure 3D input
        if features.ndim == 2:
            features = features[np.newaxis, ...]
        
        # Convert to tensor
        x = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        # Encode
        self.model.eval()
        with torch.no_grad():
            state_vector = self.model.encode(x)
        
        return state_vector.cpu().numpy()
    
    def predict(self, features: np.ndarray) -> Dict:
        """
        Make prediction from features.
        
        Args:
            features: Input features (batch, seq_length, n_features)
            
        Returns:
            Dict with signal, confidence, market_state, predicted_direction
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, returning placeholder prediction")
            return {
                "signal": np.array([0.0]),
                "confidence": np.array([0.5]),
                "market_state": np.random.randn(1, self.config.output_dim),
                "predicted_direction": np.array([0]),
                "timestamp": None
            }
        
        # Ensure 3D input
        if features.ndim == 2:
            features = features[np.newaxis, ...]
        
        # Convert to tensor
        x = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        # Predict
        result = self.model.predict(x)
        result['timestamp'] = None
        
        return result
        
    def get_architecture_summary(self) -> str:
        """Get model architecture summary."""
        if TORCH_AVAILABLE and self.model:
            n_params = self.model.get_num_params()
        else:
            n_params = self._estimate_params()
            
        return f"""
CORE 1: Quant iTransformer
==========================
Architecture: iTransformer (ICLR 2024)
Innovation: Attention ACROSS variates (not time)

Input: (batch, {self.config.seq_length}, {self.config.n_features})
Variate Embedding: {self.config.d_model} per feature
Encoder Blocks: {self.config.n_layers} x iTransformer
Attention Heads: {self.config.n_heads}
Output: {self.config.output_dim}-dim Market State Vector

Parameters: {n_params:,}
Device: {self.device}
Loaded: {self.is_loaded}
"""
    
    def _estimate_params(self) -> int:
        """Estimate number of parameters."""
        embed_params = self.config.seq_length * self.config.d_model * self.config.n_features
        attn_params = 4 * self.config.d_model ** 2 * self.config.n_layers
        ff_params = 2 * self.config.d_model * self.config.d_ff * self.config.n_layers
        output_params = self.config.d_model * self.config.n_features * self.config.d_ff
        
        return embed_params + attn_params + ff_params + output_params


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize
    model = QuantTransformer()
    print(model.get_architecture_summary())
    
    # Test with dummy data
    dummy_input = np.random.randn(2, 60, 30).astype(np.float32)
    result = model.predict(dummy_input)
    
    print(f"\nTest prediction:")
    print(f"Signal: {result['signal']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Direction: {result['predicted_direction']}")
    print(f"State vector shape: {result['state_vector'].shape}")
    
    if TORCH_AVAILABLE:
        print(f"\nPyTorch model test:")
        x = torch.randn(2, 60, 30)
        model.model.eval()
        with torch.no_grad():
            out = model.model(x)
        print(f"State vector: {out['state_vector'].shape}")
        print(f"Logits: {out['logits'].shape}")
