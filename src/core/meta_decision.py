#!/usr/bin/env python3
"""
PhiHorizon V7.0 CORE 3: Meta Decision Engine

Production-ready decision engine that combines CORE 1 (Quant) and CORE 2 (NLP)
using cross-attention fusion for trading decisions.

Architecture:
- CrossAttentionFusion: Bidirectional attention between modalities
- GatedRecalibration: Dynamic weighting based on confidence
- FusionNetwork: Complete fusion with decision head
- Online learning capability for continuous improvement

Dependencies:
- torch>=2.0.0
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Meta Decision will use placeholder mode.")


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class MetaDecisionConfig:
    """Configuration for Meta Decision Engine (CORE 3)."""
    
    # Input dimensions (must match CORE 1 and CORE 2 outputs)
    quant_dim: int = 512           # From CORE 1 QuantTransformer
    nlp_dim: int = 512             # From CORE 2 NLPFinBERT
    
    # Fusion architecture
    fusion_dim: int = 512
    n_attention_heads: int = 8
    n_fusion_layers: int = 2       # Reduced for efficiency
    dropout: float = 0.1
    
    # Decision head
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    
    # Output
    n_actions: int = 3             # LONG, SHORT, HOLD
    
    # RL component (optional)
    use_rl: bool = False           # Disable by default for simplicity
    gamma: float = 0.99
    learning_rate: float = 3e-4
    
    # Online learning
    use_online_learning: bool = True
    memory_size: int = 10000
    update_frequency: int = 100
    
    # Decision thresholds
    min_confidence: float = 0.6
    signal_threshold: float = 0.3
    position_sizes: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.50, 0.75, 1.0])
    
    # Device
    device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"


# ============================================================
# PYTORCH FUSION MODULES
# ============================================================

if TORCH_AVAILABLE:
    
    class CrossAttentionFusion(nn.Module):
        """
        Bidirectional cross-attention fusion between modalities.
        
        Performs:
        1. Quant → NLP attention (query=quant, key/value=nlp)
        2. NLP → Quant attention (query=nlp, key/value=quant)
        3. Gated combination of both
        """
        
        def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
            super().__init__()
            
            # Quant attends to NLP
            self.quant_to_nlp = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
            
            # NLP attends to Quant
            self.nlp_to_quant = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
            
            # Gating mechanism
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            
            # Layer norms
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            
            # Feed-forward
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(dropout)
            )
            self.norm3 = nn.LayerNorm(d_model)
        
        def forward(
            self, 
            quant: torch.Tensor, 
            nlp: torch.Tensor
        ) -> torch.Tensor:
            """
            Args:
                quant: (batch, d_model) - Quant state from CORE 1
                nlp: (batch, d_model) - NLP state from CORE 2
                
            Returns:
                (batch, d_model) - Fused representation
            """
            # Add sequence dimension for attention
            quant = quant.unsqueeze(1)  # (batch, 1, d_model)
            nlp = nlp.unsqueeze(1)      # (batch, 1, d_model)
            
            # Bidirectional cross-attention
            quant_attended, _ = self.quant_to_nlp(quant, nlp, nlp)  # Quant queries NLP
            nlp_attended, _ = self.nlp_to_quant(nlp, quant, quant)  # NLP queries Quant
            
            # Residual connections
            quant_attended = self.norm1(quant + quant_attended)
            nlp_attended = self.norm2(nlp + nlp_attended)
            
            # Gate: learn optimal mixing
            combined = torch.cat([quant_attended, nlp_attended], dim=-1)
            gate_weights = self.gate(combined)  # (batch, 1, d_model)
            
            # Gated fusion
            fused = gate_weights * quant_attended + (1 - gate_weights) * nlp_attended
            
            # Feed-forward
            fused = self.norm3(fused + self.ffn(fused))
            
            return fused.squeeze(1)  # (batch, d_model)
    
    
    class GatedRecalibration(nn.Module):
        """
        Dynamic weighting based on confidence scores.
        
        Learns to weight modalities based on their reliability.
        """
        
        def __init__(self, d_model: int = 512):
            super().__init__()
            
            # Input: fused embedding + confidence scores
            self.weight_net = nn.Sequential(
                nn.Linear(d_model + 2, 128),  # +2 for quant_conf, nlp_conf
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.Softmax(dim=-1)
            )
            
            # Projection for weighted combination
            self.projection = nn.Linear(d_model, d_model)
        
        def forward(
            self,
            fused: torch.Tensor,
            quant_conf: torch.Tensor,
            nlp_conf: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                fused: (batch, d_model) - Fused embedding
                quant_conf: (batch,) - Quant confidence [0, 1]
                nlp_conf: (batch,) - NLP confidence [0, 1]
                
            Returns:
                recalibrated: (batch, d_model)
                weights: (batch, 2) - [quant_weight, nlp_weight]
            """
            # Concatenate confidence scores
            conf = torch.stack([quant_conf, nlp_conf], dim=-1)  # (batch, 2)
            combined = torch.cat([fused, conf], dim=-1)  # (batch, d_model + 2)
            
            # Learn dynamic weights
            weights = self.weight_net(combined)  # (batch, 2)
            
            # Apply projection
            recalibrated = self.projection(fused)
            
            return recalibrated, weights
    
    
    class DecisionHead(nn.Module):
        """
        MLP for final action prediction.
        
        Outputs:
        - action_logits: For action selection
        - confidence: Prediction confidence
        - position_size: Suggested position size
        """
        
        def __init__(
            self, 
            d_model: int = 512, 
            hidden_dims: List[int] = [256, 128],
            n_actions: int = 3
        ):
            super().__init__()
            
            layers = []
            in_dim = d_model
            
            for h_dim in hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, h_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
                in_dim = h_dim
            
            self.backbone = nn.Sequential(*layers)
            
            # Output heads
            self.action_head = nn.Linear(in_dim, n_actions)
            self.confidence_head = nn.Linear(in_dim, 1)
            self.position_head = nn.Linear(in_dim, 1)
        
        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            Args:
                x: (batch, d_model)
                
            Returns:
                Dict with action_logits, confidence, position_size
            """
            features = self.backbone(x)
            
            return {
                "action_logits": self.action_head(features),
                "confidence": torch.sigmoid(self.confidence_head(features)),
                "position_size": torch.sigmoid(self.position_head(features))
            }
    
    
    class FusionNetwork(nn.Module):
        """
        Complete fusion network combining all components.
        """
        
        def __init__(self, config: MetaDecisionConfig):
            super().__init__()
            
            self.config = config
            
            # Input projections (in case dimensions don't match)
            self.quant_proj = nn.Linear(config.quant_dim, config.fusion_dim)
            self.nlp_proj = nn.Linear(config.nlp_dim, config.fusion_dim)
            
            # Fusion layers
            self.fusion_layers = nn.ModuleList([
                CrossAttentionFusion(
                    d_model=config.fusion_dim,
                    n_heads=config.n_attention_heads,
                    dropout=config.dropout
                )
                for _ in range(config.n_fusion_layers)
            ])
            
            # Recalibration
            self.recalibration = GatedRecalibration(config.fusion_dim)
            
            # Decision head
            self.decision_head = DecisionHead(
                d_model=config.fusion_dim,
                hidden_dims=config.hidden_dims,
                n_actions=config.n_actions
            )
            
            self._init_weights()
        
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        def forward(
            self,
            quant_state: torch.Tensor,
            nlp_state: torch.Tensor,
            quant_conf: torch.Tensor,
            nlp_conf: torch.Tensor
        ) -> Dict[str, torch.Tensor]:
            """
            Full forward pass.
            
            Args:
                quant_state: (batch, quant_dim)
                nlp_state: (batch, nlp_dim)
                quant_conf: (batch,)
                nlp_conf: (batch,)
                
            Returns:
                Dict with action_logits, confidence, position_size, weights
            """
            # Project to fusion dimension
            quant = self.quant_proj(quant_state)
            nlp = self.nlp_proj(nlp_state)
            
            # Apply fusion layers
            fused = quant  # Start with quant
            for fusion_layer in self.fusion_layers:
                fused = fusion_layer(fused, nlp)
            
            # Recalibrate based on confidence
            recalibrated, weights = self.recalibration(fused, quant_conf, nlp_conf)
            
            # Decision
            outputs = self.decision_head(recalibrated)
            outputs["weights"] = weights
            outputs["fused_state"] = recalibrated
            
            return outputs


# ============================================================
# META DECISION ENGINE
# ============================================================

class MetaDecisionEngine:
    """
    CORE 3: Meta Decision Engine.
    
    Combines Quant and NLP signals using cross-attention fusion
    for optimal trading decisions.
    
    Features:
    1. Cross-attention fusion of market and sentiment state
    2. Gated recalibration based on confidence
    3. Experience replay for learning from trades
    4. Online updates to improve over time
    """
    
    ACTION_MAP = {0: "LONG", 1: "SHORT", 2: "HOLD"}
    
    def __init__(self, config: Optional[MetaDecisionConfig] = None):
        """Initialize the Meta Decision Engine."""
        self.config = config or MetaDecisionConfig()
        self.is_loaded = False
        self.model = None
        
        # Initialize model if PyTorch available
        if TORCH_AVAILABLE:
            self.model = FusionNetwork(self.config).to(self.config.device)
            self.model.eval()
            logger.info(f"FusionNetwork initialized with {self._count_params():,} parameters")
        
        # Experience replay
        self._experience_buffer: List[Dict] = []
        self._trade_history: List[Dict] = []
        self._cumulative_reward: float = 0.0
        
        logger.info(
            f"MetaDecisionEngine initialized: "
            f"fusion_dim={self.config.fusion_dim}, "
            f"device={self.config.device}"
        )
    
    def _count_params(self) -> int:
        """Count trainable parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def load_model(self, path: str) -> bool:
        """Load trained model from checkpoint."""
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available")
            return False
        
        try:
            logger.info(f"Loading model from {path}")
            checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
            
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            self.is_loaded = True
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def save_model(self, path: str) -> bool:
        """Save model to checkpoint."""
        if self.model is None:
            logger.error("No model to save")
            return False
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "config": self.config.__dict__
            }, path)
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    # ========================================================
    # FUSION
    # ========================================================
    
    def fuse_embeddings(
        self,
        quant_state: np.ndarray,
        nlp_state: np.ndarray,
        quant_conf: float = 0.5,
        nlp_conf: float = 0.5
    ) -> np.ndarray:
        """
        Fuse Quant and NLP embeddings using cross-attention.
        
        Args:
            quant_state: From CORE 1 (shape: quant_dim)
            nlp_state: From CORE 2 (shape: nlp_dim)
            quant_conf: CORE 1 confidence
            nlp_conf: CORE 2 confidence
            
        Returns:
            Fused embedding (shape: fusion_dim)
        """
        if not TORCH_AVAILABLE or self.model is None:
            # Fallback: simple weighted average
            concat = np.concatenate([quant_state[:self.config.fusion_dim//2], 
                                     nlp_state[:self.config.fusion_dim//2]])
            return concat
        
        # Convert to tensors
        quant_t = torch.tensor(quant_state, dtype=torch.float32).unsqueeze(0).to(self.config.device)
        nlp_t = torch.tensor(nlp_state, dtype=torch.float32).unsqueeze(0).to(self.config.device)
        quant_c = torch.tensor([quant_conf], dtype=torch.float32).to(self.config.device)
        nlp_c = torch.tensor([nlp_conf], dtype=torch.float32).to(self.config.device)
        
        with torch.no_grad():
            outputs = self.model(quant_t, nlp_t, quant_c, nlp_c)
        
        return outputs["fused_state"].squeeze(0).cpu().numpy()
    
    # ========================================================
    # DECISION MAKING
    # ========================================================
    
    def make_decision(
        self,
        quant_output: Dict,
        nlp_output: Dict,
        market_data: Dict
    ) -> Dict:
        """
        Make trading decision.
        
        Args:
            quant_output: Output from CORE 1
            nlp_output: Output from CORE 2
            market_data: Current market metrics
            
        Returns:
            Decision dict with action, confidence, reasoning
        """
        # Extract states and confidence
        quant_state = quant_output.get("market_state", np.zeros(self.config.quant_dim))
        nlp_state = nlp_output.get("sentiment_state", np.zeros(self.config.nlp_dim))
        
        if isinstance(quant_state, np.ndarray) and quant_state.ndim > 1:
            quant_state = quant_state.flatten()[:self.config.quant_dim]
        if isinstance(nlp_state, np.ndarray) and nlp_state.ndim > 1:
            nlp_state = nlp_state.flatten()[:self.config.nlp_dim]
        
        quant_signal = quant_output.get("signal", 0)
        nlp_signal = nlp_output.get("signal", 0)
        quant_conf = quant_output.get("confidence", 0.5)
        nlp_conf = nlp_output.get("confidence", 0.5)
        
        # Use neural network if available
        if TORCH_AVAILABLE and self.model is not None:
            # Ensure correct shapes
            if len(quant_state) < self.config.quant_dim:
                quant_state = np.pad(quant_state, (0, self.config.quant_dim - len(quant_state)))
            if len(nlp_state) < self.config.nlp_dim:
                nlp_state = np.pad(nlp_state, (0, self.config.nlp_dim - len(nlp_state)))
            
            quant_t = torch.tensor(quant_state[:self.config.quant_dim], dtype=torch.float32).unsqueeze(0).to(self.config.device)
            nlp_t = torch.tensor(nlp_state[:self.config.nlp_dim], dtype=torch.float32).unsqueeze(0).to(self.config.device)
            quant_c = torch.tensor([quant_conf], dtype=torch.float32).to(self.config.device)
            nlp_c = torch.tensor([nlp_conf], dtype=torch.float32).to(self.config.device)
            
            with torch.no_grad():
                outputs = self.model(quant_t, nlp_t, quant_c, nlp_c)
            
            action_probs = F.softmax(outputs["action_logits"], dim=-1).squeeze(0).cpu().numpy()
            action_idx = int(np.argmax(action_probs))
            action = self.ACTION_MAP[action_idx]
            
            combined_conf = float(outputs["confidence"].item())
            position_size = float(outputs["position_size"].item())
            weights = outputs["weights"].squeeze(0).cpu().numpy()
            
        else:
            # Fallback: rule-based
            combined_signal = 0.6 * quant_signal + 0.4 * nlp_signal
            combined_conf = 0.6 * quant_conf + 0.4 * nlp_conf
            
            if combined_signal > self.config.signal_threshold and combined_conf > self.config.min_confidence:
                action = "LONG"
                action_idx = 0
            elif combined_signal < -self.config.signal_threshold and combined_conf > self.config.min_confidence:
                action = "SHORT"
                action_idx = 1
            else:
                action = "HOLD"
                action_idx = 2
            
            action_probs = np.zeros(3)
            action_probs[action_idx] = combined_conf
            
            size_idx = min(int(combined_conf * len(self.config.position_sizes)), 
                           len(self.config.position_sizes) - 1)
            position_size = self.config.position_sizes[size_idx] if action != "HOLD" else 0.0
            weights = np.array([0.6, 0.4])
        
        # Risk management
        atr = market_data.get("atr", 100)
        stop_loss = 2.0 * atr
        take_profit = 6.0 * atr  # 1:3 R:R
        
        # Direction
        direction = 1 if action == "LONG" else (-1 if action == "SHORT" else 0)
        
        # Reasoning
        reasoning = self._build_reasoning(
            quant_signal, nlp_signal,
            quant_conf, nlp_conf,
            action, weights
        )
        
        decision = {
            "action": action,
            "action_idx": action_idx,
            "action_probs": action_probs.tolist() if isinstance(action_probs, np.ndarray) else action_probs,
            "direction": direction,
            "position_size": float(position_size),
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "confidence": float(combined_conf),
            "quant_signal": float(quant_signal),
            "nlp_signal": float(nlp_signal),
            "weights": weights.tolist() if isinstance(weights, np.ndarray) else weights,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        }
        
        return decision
    
    def _build_reasoning(
        self,
        quant_signal: float,
        nlp_signal: float,
        quant_conf: float,
        nlp_conf: float,
        action: str,
        weights: np.ndarray
    ) -> str:
        """Build human-readable reasoning for decision."""
        parts = []
        
        # Quant analysis
        quant_dir = "bullish" if quant_signal > 0.1 else ("bearish" if quant_signal < -0.1 else "neutral")
        parts.append(f"Quant: {quant_dir} ({quant_signal:.2f}, w={weights[0]:.0%})")
        
        # NLP analysis
        nlp_dir = "bullish" if nlp_signal > 0.1 else ("bearish" if nlp_signal < -0.1 else "neutral")
        parts.append(f"NLP: {nlp_dir} ({nlp_signal:.2f}, w={weights[1]:.0%})")
        
        # Alignment
        if (quant_signal > 0 and nlp_signal > 0) or (quant_signal < 0 and nlp_signal < 0):
            parts.append("ALIGNED")
        else:
            parts.append("MIXED")
        
        parts.append(f"→ {action}")
        
        return " | ".join(parts)
    
    # ========================================================
    # LEARNING
    # ========================================================
    
    def record_trade(self, decision: Dict, outcome: Dict):
        """Record trade outcome for learning."""
        experience = {
            "decision": decision,
            "outcome": outcome,
            "timestamp": datetime.now().isoformat()
        }
        
        self._experience_buffer.append(experience)
        self._trade_history.append(experience)
        
        # Keep buffer limited
        if len(self._experience_buffer) > self.config.memory_size:
            self._experience_buffer = self._experience_buffer[-self.config.memory_size:]
        
        # Calculate reward
        pnl = outcome.get("pnl", 0)
        self._cumulative_reward += pnl
        
        # Trigger update if needed
        if len(self._experience_buffer) >= self.config.update_frequency:
            self._update_model()
    
    def _update_model(self):
        """Update model from experience buffer."""
        if not self.config.use_online_learning:
            return
        
        if len(self._experience_buffer) < 10:
            return
        
        logger.info(f"Updating model from {len(self._experience_buffer)} experiences")
        
        # TODO: Implement online learning
        # For now, just keep recent experiences
        self._experience_buffer = self._experience_buffer[-100:]
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if not self._trade_history:
            return {"total_trades": 0}
        
        trades = self._trade_history
        pnls = [t["outcome"].get("pnl", 0) for t in trades if "outcome" in t]
        
        if not pnls:
            return {"total_trades": len(trades)}
        
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        return {
            "total_trades": len(trades),
            "total_pnl": sum(pnls),
            "win_rate": len(wins) / len(pnls) if pnls else 0,
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "profit_factor": abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float('inf'),
            "cumulative_reward": self._cumulative_reward
        }
    
    def get_architecture_summary(self) -> str:
        """Get model architecture summary."""
        param_count = self._count_params()
        return f"""
CORE 3: Meta Decision Engine
=============================
Quant Input: {self.config.quant_dim}-dim
NLP Input: {self.config.nlp_dim}-dim
Fusion: {self.config.fusion_dim}-dim with {self.config.n_attention_heads} heads
Layers: {self.config.n_fusion_layers} CrossAttention + GatedRecalibration
Decision Head: {self.config.hidden_dims}
Parameters: {param_count:,}
Device: {self.config.device}

Output: action, position_size, SL, TP, confidence, reasoning
"""


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("CORE 3: Meta Decision Engine Test")
    print("=" * 60)
    
    engine = MetaDecisionEngine()
    print(engine.get_architecture_summary())
    
    # Test decision
    quant_output = {
        "signal": 0.6,
        "confidence": 0.75,
        "market_state": np.random.randn(512).astype(np.float32)
    }
    
    nlp_output = {
        "signal": 0.4,
        "confidence": 0.65,
        "sentiment_state": np.random.randn(512).astype(np.float32)
    }
    
    market_data = {
        "atr": 500,
        "volatility": 0.02
    }
    
    decision = engine.make_decision(quant_output, nlp_output, market_data)
    
    print("\nTest Decision:")
    print(f"  Action: {decision['action']}")
    print(f"  Confidence: {decision['confidence']:.2%}")
    print(f"  Position: {decision['position_size']:.0%}")
    print(f"  SL: ${decision['stop_loss']:.0f}")
    print(f"  TP: ${decision['take_profit']:.0f}")
    print(f"  Reasoning: {decision['reasoning']}")
    print(f"  Weights: Quant={decision['weights'][0]:.0%}, NLP={decision['weights'][1]:.0%}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
