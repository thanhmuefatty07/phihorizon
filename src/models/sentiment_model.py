#!/usr/bin/env python3
"""
PhiHorizon V6.1 Sentiment Model Wrapper

Wraps the trained LSTM model for inference in the paper trading bot.

Model: V6.1 LSTM (18 features, 51.39% accuracy)
Usage: Filter G - blocks trades when sentiment confidence is low

Quality Checklist:
[x] Docstring for module
[x] Docstring for all functions
[x] Type hints on parameters
[x] Error handling (try/except)
[x] Logging (INFO level)
[x] Model loading with fallback
[x] Feature validation

Note: This is for SIMULATION/BACKTESTING only, NOT live trading.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# TensorFlow import with fallback
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. Sentiment model will be disabled.")


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class SentimentModelConfig:
    """Configuration for Sentiment LSTM Model.
    
    V6.1.2 ULTIMATE: Changed from FILTER to CONFIRMATION mode.
    Research: 51% accuracy model should scale positions, not block trades.
    - Never blocks trades (misses opportunities)
    - Instead adjusts position size based on confidence
    """
    
    # Model paths (relative to project root)
    model_path: str = "models/sentiment/v61_lstm_best.h5"
    scaler_path: str = "models/sentiment/v61_scaler.json"
    metadata_path: str = "models/sentiment/v61_training_metadata.json"
    
    # Sequence settings
    seq_length: int = 7  # 7-day lookback
    
    # V6.1.2 CONFIRMATION MODE THRESHOLDS
    # Used to scale position size, NOT to block trades
    confirmation_bullish_threshold: float = 0.60   # Above = increase position
    confirmation_bearish_threshold: float = 0.40   # Below = decrease position
    
    # Position size scaling factors
    bullish_position_scale: float = 1.2   # +20% position on bullish
    bearish_position_scale: float = 0.8   # -20% position on bearish
    neutral_position_scale: float = 1.0   # Normal position
    
    # Feature settings
    expected_features: int = 18


# ============================================================
# SENTIMENT MODEL
# ============================================================

class SentimentModel:
    """
    V6.1 LSTM Sentiment Model for trade filtering.
    
    Provides:
    - Prediction of next-day direction probability
    - Confidence-based trade filtering
    - Integration with paper trading bot
    """
    
    def __init__(
        self,
        config: Optional[SentimentModelConfig] = None,
        project_root: Optional[str] = None
    ):
        """
        Initialize the Sentiment Model.
        
        Args:
            config: Model configuration
            project_root: Path to project root (for model files)
        """
        self.config = config or SentimentModelConfig()
        
        # Determine project root
        if project_root:
            self.project_root = Path(project_root)
        else:
            # Default: assume we're in src/models/
            self.project_root = Path(__file__).parent.parent.parent
        
        # Initialize state
        self.model = None
        self.scaler_mean: Optional[Dict[str, float]] = None
        self.scaler_std: Optional[Dict[str, float]] = None
        self.feature_cols: List[str] = []
        self.is_loaded = False
        self._sequence_buffer: List[np.ndarray] = []
        
        # Try to load model
        self._load_model()
    
    def _load_model(self):
        """Load the LSTM model and scaler."""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available, model disabled")
            return
        
        try:
            # Load model
            model_path = self.project_root / self.config.model_path
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                return
            
            self.model = load_model(str(model_path), compile=False)
            logger.info(f"Loaded model from {model_path}")
            
            # Load scaler
            scaler_path = self.project_root / self.config.scaler_path
            if scaler_path.exists():
                with open(scaler_path, 'r') as f:
                    scaler_data = json.load(f)
                self.scaler_mean = scaler_data['mean']
                self.scaler_std = scaler_data['std']
                self.feature_cols = list(self.scaler_mean.keys())
                logger.info(f"Loaded scaler with {len(self.feature_cols)} features")
            else:
                logger.warning(f"Scaler file not found: {scaler_path}")
            
            # Load metadata for info
            meta_path = self.project_root / self.config.metadata_path
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Model version: {metadata.get('version', 'unknown')}")
                logger.info(f"Model accuracy: {metadata.get('test_accuracy', 0):.2%}")
            
            self.is_loaded = True
            logger.info("SentimentModel loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
    
    # ================================================================
    # FEATURE PREPARATION
    # ================================================================
    
    def prepare_features(self, data: Dict[str, float]) -> Optional[np.ndarray]:
        """
        Prepare and normalize features for model input.
        
        Args:
            data: Dict of feature_name -> value
            
        Returns:
            Normalized feature array, or None if invalid
        """
        if not self.is_loaded or not self.scaler_mean:
            return None
        
        try:
            # Extract features in correct order
            features = []
            for col in self.feature_cols:
                if col in data:
                    value = data[col]
                else:
                    # Use 0 for missing features (will be normalized to mean)
                    value = self.scaler_mean.get(col, 0)
                    logger.debug(f"Missing feature {col}, using mean")
                
                features.append(float(value))
            
            features = np.array(features)
            
            # Normalize
            mean = np.array([self.scaler_mean[c] for c in self.feature_cols])
            std = np.array([self.scaler_std[c] for c in self.feature_cols])
            std = np.where(std == 0, 1, std)  # Avoid division by zero
            
            normalized = (features - mean) / std
            
            return normalized
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return None
    
    def update_sequence_buffer(self, features: np.ndarray):
        """
        Update the rolling sequence buffer.
        
        Args:
            features: New feature array to add
        """
        self._sequence_buffer.append(features)
        
        # Keep only last seq_length samples
        if len(self._sequence_buffer) > self.config.seq_length:
            self._sequence_buffer = self._sequence_buffer[-self.config.seq_length:]
    
    def get_sequence(self) -> Optional[np.ndarray]:
        """
        Get current sequence for prediction.
        
        Returns:
            Sequence array (seq_length, n_features), or None if not enough data
        """
        if len(self._sequence_buffer) < self.config.seq_length:
            logger.debug(f"Sequence buffer not full: {len(self._sequence_buffer)}/{self.config.seq_length}")
            return None
        
        sequence = np.array(self._sequence_buffer[-self.config.seq_length:])
        return sequence.reshape(1, self.config.seq_length, -1)
    
    # ================================================================
    # PREDICTION
    # ================================================================
    
    def predict(self, sequence: Optional[np.ndarray] = None) -> Optional[float]:
        """
        Predict probability of next-day upward movement.
        
        Args:
            sequence: Optional input sequence (uses buffer if None)
            
        Returns:
            Probability (0-1), or None if prediction failed
        """
        if not self.is_loaded or self.model is None:
            return None
        
        try:
            if sequence is None:
                sequence = self.get_sequence()
            
            if sequence is None:
                return None
            
            # Predict
            prediction = self.model.predict(sequence, verbose=0)
            probability = float(prediction[0][0])
            
            logger.debug(f"Predicted probability: {probability:.3f}")
            
            return probability
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def predict_from_features(self, data: Dict[str, float]) -> Optional[float]:
        """
        Convenience method: prepare features and predict.
        
        Args:
            data: Dict of feature_name -> value
            
        Returns:
            Probability (0-1), or None if failed
        """
        features = self.prepare_features(data)
        if features is None:
            return None
        
        self.update_sequence_buffer(features)
        return self.predict()
    
    # ================================================================
    # FILTER DECISION
    # ================================================================
    
    def classify_confidence(self, probability: float) -> str:
        """
        Classify prediction confidence level for position scaling.
        
        V6.1.2 CONFIRMATION MODE: Used to scale position, not block.
        
        Args:
            probability: Model prediction (0-1)
            
        Returns:
            "bullish", "bearish", or "neutral"
        """
        if probability >= self.config.confirmation_bullish_threshold:
            return "bullish"
        elif probability <= self.config.confirmation_bearish_threshold:
            return "bearish"
        else:
            return "neutral"
    
    def get_position_scale(
        self,
        probability: Optional[float] = None,
        features: Optional[Dict[str, float]] = None
    ) -> Tuple[float, str]:
        """
        V6.1.2 ULTIMATE: Get position size scaling factor.
        
        CONFIRMATION MODE - Never blocks trades, only adjusts position size:
        - Bullish (>=0.60): Scale up 20% (1.2x)
        - Bearish (<=0.40): Scale down 20% (0.8x)
        - Neutral: Normal size (1.0x)
        
        Research: 51% accuracy model should not block trades, but can
        provide edge through position sizing adjustments.
        
        Args:
            probability: Pre-computed probability
            features: Features dict (will compute probability if not provided)
            
        Returns:
            Tuple of (scale_factor, reason)
        """
        # Get probability
        if probability is None:
            if features is not None:
                probability = self.predict_from_features(features)
            else:
                probability = self.predict()
        
        # Handle missing prediction
        if probability is None:
            return self.config.neutral_position_scale, "Model unavailable - normal size"
        
        # CONFIRMATION MODE: Scale position, never block
        classification = self.classify_confidence(probability)
        
        if classification == "bullish":
            return self.config.bullish_position_scale, f"Bullish ({probability:.1%}) - scale up"
        elif classification == "bearish":
            return self.config.bearish_position_scale, f"Bearish ({probability:.1%}) - scale down"
        else:
            return self.config.neutral_position_scale, f"Neutral ({probability:.1%}) - normal size"
    
    def should_allow_trade(
        self,
        probability: Optional[float] = None,
        features: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, str]:
        """
        V6.1.2: ALWAYS allows trade (confirmation mode).
        
        Kept for backward compatibility, but always returns True.
        Use get_position_scale() instead for position sizing.
        
        Args:
            probability: Pre-computed probability
            features: Features dict
            
        Returns:
            Always (True, reason) - never blocks
        """
        scale, reason = self.get_position_scale(probability, features)
        return True, f"Confirmed (scale={scale:.1f}x) - {reason}"
    
    # ================================================================
    # STATUS
    # ================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get model status and info."""
        return {
            "is_loaded": self.is_loaded,
            "tf_available": TF_AVAILABLE,
            "n_features": len(self.feature_cols),
            "seq_length": self.config.seq_length,
            "buffer_size": len(self._sequence_buffer),
            "buffer_ready": len(self._sequence_buffer) >= self.config.seq_length,
        }


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_sentiment_model(project_root: str = None) -> SentimentModel:
    """Create a SentimentModel instance."""
    return SentimentModel(project_root=project_root)


# ============================================================
# MAIN (for testing)
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "=" * 60)
    print("SENTIMENT MODEL TEST")
    print("=" * 60)
    
    # Find project root
    project_root = Path(__file__).parent.parent.parent
    print(f"\nProject root: {project_root}")
    
    # Create model
    model = SentimentModel(project_root=str(project_root))
    
    # Check status
    print("\n📊 Model Status:")
    status = model.get_status()
    for k, v in status.items():
        print(f"   {k}: {v}")
    
    if not model.is_loaded:
        print("\n❌ Model not loaded, exiting")
        exit(1)
    
    # Test with dummy data
    print("\n🧪 Testing with dummy features...")
    
    # Simulate 7 days of data
    for day in range(7):
        dummy_features = {
            'fear_greed': 50 + day * 2,
            'fg_ma7': 48 + day,
            'fg_change': 1.0,
            'google_trends': 45 + day,
            'trends_ma7': 43,
            'trends_change_7d': 0.1,
            'btc_dominance': 52,
            'dominance_change_7d': 0.01,
            'stablecoin_ratio': 0.08,
            'hash_rate_norm': 0.5,
            'hash_rate_change_7d': 0.02,
            'hash_rate_trend': 1,
            'btc_return_1d': 0.01,
            'btc_return_7d': 0.05,
            'btc_volatility': 0.03,
            'sentiment_composite': 0.55,
            'risk_score': 0.4,
            'momentum_score': 0.02,
        }
        
        features = model.prepare_features(dummy_features)
        model.update_sequence_buffer(features)
        print(f"   Day {day + 1}: Buffer size = {len(model._sequence_buffer)}")
    
    # Predict
    print("\n🎯 Making prediction...")
    probability = model.predict()
    
    if probability is not None:
        print(f"✅ Prediction: {probability:.2%} bullish")
        
        classification = model.classify_confidence(probability)
        print(f"   Classification: {classification}")
        
        allow, reason = model.should_allow_trade(probability)
        print(f"   Allow trade: {allow}")
        print(f"   Reason: {reason}")
    else:
        print("❌ Prediction failed")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
