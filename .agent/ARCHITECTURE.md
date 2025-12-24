# 📐 PHIHORIZON - TECHNICAL ARCHITECTURE

**Version:** V7.0  
**Last Update:** Dec 24, 2025

---

## 🏗️ SYSTEM ARCHITECTURE

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           PHIHORIZON V7.0                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         DATA LAYER (14 Sources)                         │ │
│  ├─────────────────────────────────────────────────────────────────────────┤ │
│  │ Exchange:   OHLCV, Funding Rate, OI, Long/Short Ratio (Binance)        │ │
│  │ On-Chain:   Hash Rate, Whale Netflow, Reserve, Active Addresses        │ │
│  │ Sentiment:  Fear & Greed, Google Trends                                │ │
│  │ News:       CoinDesk, CoinTelegraph, Decrypt, CryptoPanic              │ │
│  │ Social:     Twitter, Reddit, Telegram (needs API)                      │ │
│  │ Macro:      DXY, Fed Rates, CPI (planned)                              │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                    ┌───────────────┴───────────────┐                        │
│                    ▼                               ▼                        │
│  ┌─────────────────────────────┐   ┌─────────────────────────────┐         │
│  │       ML GUARD 1            │   │       ML GUARD 2            │         │
│  │      QuantGuard             │   │       NLPGuard              │         │
│  ├─────────────────────────────┤   ├─────────────────────────────┤         │
│  │ • Z-score outlier detection │   │ • Spam detection            │         │
│  │ • IQR anomaly filtering     │   │ • Source credibility        │         │
│  │ • Range validation          │   │ • Duplicate removal         │         │
│  │ • Freshness checking        │   │ • Relevance filtering       │         │
│  │ • Missing data handling     │   │ • Language detection        │         │
│  └──────────────┬──────────────┘   └──────────────┬──────────────┘         │
│                 │                                  │                        │
│                 ▼                                  ▼                        │
│  ┌─────────────────────────────┐   ┌─────────────────────────────┐         │
│  │        CORE 1               │   │        CORE 2               │         │
│  │    QuantTransformer         │   │      NLPFinBERT             │         │
│  ├─────────────────────────────┤   ├─────────────────────────────┤         │
│  │ • 8-layer Transformer       │   │ • ProsusAI/finbert base     │         │
│  │ • 30 market features        │   │ • Fine-tuned on crypto news │         │
│  │ • 60-day lookback           │   │ • Sentiment classification  │         │
│  │ • Output: 512-dim vector    │   │ • Output: 768-dim vector    │         │
│  │ STATUS: PLACEHOLDER ⚠️      │   │ STATUS: PLACEHOLDER ⚠️      │         │
│  └──────────────┬──────────────┘   └──────────────┬──────────────┘         │
│                 │                                  │                        │
│                 └───────────────┬──────────────────┘                        │
│                                 ▼                                           │
│                 ┌─────────────────────────────┐                             │
│                 │       ML GUARD 3            │                             │
│                 │      FusionGuard            │                             │
│                 ├─────────────────────────────┤                             │
│                 │ • Signal conflict detection │                             │
│                 │ • Market regime classification│                           │
│                 │ • Confidence calibration    │                             │
│                 │ • Risk assessment           │                             │
│                 └──────────────┬──────────────┘                             │
│                                ▼                                            │
│                 ┌─────────────────────────────┐                             │
│                 │        CORE 3               │                             │
│                 │    MetaDecision Engine      │                             │
│                 ├─────────────────────────────┤                             │
│                 │ • Cross-attention fusion    │                             │
│                 │ • Action: BUY/SELL/HOLD     │                             │
│                 │ • Confidence: 0-1           │                             │
│                 │ • RL online learning        │                             │
│                 │ STATUS: PLACEHOLDER ⚠️      │                             │
│                 └──────────────┬──────────────┘                             │
│                                │                                            │
│                    ┌───────────┴───────────┐                               │
│                    ▼                       ▼                               │
│  ┌─────────────────────────────┐   ┌─────────────────────────────┐         │
│  │      PHI FILTER             │   │    POSITION SIZER           │         │
│  │  consciousness/metrics.py   │   │    risk/position_sizer.py   │         │
│  ├─────────────────────────────┤   ├─────────────────────────────┤         │
│  │ • IIT-based Phi metric      │   │ • Kelly Criterion           │         │
│  │ • Market integration score  │   │ • ATR-based stop loss       │         │
│  │ • Filters noisy markets     │   │ • Volatility adjustment     │         │
│  │ STATUS: COMPLETE ✅         │   │ STATUS: COMPLETE ✅         │         │
│  └─────────────────────────────┘   └─────────────────────────────┘         │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                       BACKTESTING & VALIDATION                          │ │
│  ├─────────────────────────────────────────────────────────────────────────┤ │
│  │ Walk-Forward Optimizer │ Production Backtester │ Monte Carlo Simulation │ │
│  │ STATUS: COMPLETE ✅                                                     │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 FILE STRUCTURE

```
PhiHorizon/
├── .agent/
│   ├── MEMORY.md                    # 🧠 PERMANENT MEMORY (READ THIS FIRST!)
│   └── workflows/
│       ├── phihorizon-init.md       # Load memory workflow
│       ├── maximum-effort.md        # Maximum effort protocol
│       ├── expert_team_core.md      # Expert team protocol
│       └── hybrid_cloud_dispatch.md # Cloud training dispatch
├── src/
│   ├── __init__.py                  # Package init, exports main classes
│   ├── consciousness/               # IIT Phi metrics ✅ COMPLETE
│   │   ├── __init__.py
│   │   ├── metrics.py               # PhiCalculator, IITCore (500+ lines)
│   │   └── entropy_metrics.py       # Transfer entropy, MI (300+ lines)
│   ├── core/                        # CORE models ⚠️ PLACEHOLDERS
│   │   ├── __init__.py
│   │   ├── quant_transformer.py     # CORE 1 (192 lines, placeholder)
│   │   ├── nlp_finbert.py           # CORE 2 (234 lines, placeholder)
│   │   └── meta_decision.py         # CORE 3 (351 lines, placeholder)
│   ├── guards/                      # ML Guards ✅ COMPLETE
│   │   ├── __init__.py
│   │   ├── quant_guard.py           # QuantGuard (457 lines)
│   │   └── nlp_guard.py             # NLPGuard + FusionGuard (366 lines)
│   ├── data/                        # Data Loaders ✅ COMPLETE
│   │   ├── __init__.py
│   │   ├── binance_loader.py        # OI, L/S Ratio, Funding (513 lines)
│   │   ├── onchain_loader.py        # Whale, Hash rate
│   │   ├── sentiment_loader.py      # F&G Index
│   │   ├── coingecko_loader.py      # BTC.D, MCap
│   │   ├── news_loader.py           # News aggregation
│   │   ├── social_loader.py         # Social metrics
│   │   ├── funding_loader.py        # Funding rates
│   │   ├── ccxt_loader.py           # Multi-exchange
│   │   ├── blockchain_loader.py     # Blockchain data
│   │   ├── google_trends_loader.py  # Search trends
│   │   ├── hybrid_loader.py         # Combined sources
│   │   ├── multi_source_merger.py   # Data fusion
│   │   └── data_pipeline.py         # Main pipeline (419 lines)
│   ├── strategy/                    # Trading strategies ✅ COMPLETE
│   │   ├── __init__.py
│   │   ├── base_strategy.py         # Abstract base
│   │   ├── phi_filter.py            # Phi-based filter (222 lines)
│   │   └── rsi_strategy.py          # Example strategy
│   ├── backtesting/                 # Backtesting ✅ COMPLETE
│   │   ├── __init__.py
│   │   ├── walk_forward.py          # WF Optimizer (859 lines)
│   │   └── production_backtester.py # Full backtester (1329 lines)
│   ├── risk/                        # Risk management ✅ COMPLETE
│   │   ├── __init__.py
│   │   ├── position_sizer.py        # Kelly, ATR stops (302 lines)
│   │   └── advanced_risk_manager.py # VaR, CVaR (203 lines)
│   ├── models/                      # ML wrappers
│   │   ├── __init__.py
│   │   ├── sentiment_model.py       # LSTM wrapper (396 lines)
│   │   └── xgboost_model.py
│   └── utils/                       # Utilities (14 files)
├── notebooks/
│   ├── 01_data_preparation.ipynb    ✅
│   ├── 02_sentiment_training.ipynb  ✅
│   ├── 03_lstm_training.ipynb       ✅
│   ├── 04_core1_training.ipynb      ❌ MISSING - Need to create
│   ├── 05_core2_training.ipynb      ❌ MISSING - Need to create
│   ├── 06_backtesting.ipynb         ✅
│   └── 07_paper_trading.ipynb       ✅
├── models/
│   └── sentiment/
│       └── v61_lstm_best.h5         # Trained LSTM (51.39% accuracy)
├── tests/                           # 18 test files
├── scripts/
│   └── paper_trading_bot.py         # V5.5 paper trading
├── results/
│   └── benchmark_report.json        # Current benchmarks
├── docs/
│   ├── QUICKSTART.md
│   └── BUYER_SETUP_GUIDE.md
├── pyproject.toml                   # Project config
├── requirements.txt                 # Dependencies
└── .env.example                     # Environment template
```

---

## 🔢 MODEL SPECIFICATIONS

### CORE 1: QuantTransformer
```python
config = {
    "d_model": 128,
    "n_heads": 8,
    "n_layers": 8,
    "d_ff": 512,
    "input_dim": 30,      # 30 market features
    "output_dim": 512,    # Market State Vector
    "seq_length": 60,     # 60-day lookback
    "dropout": 0.1
}
# Status: PLACEHOLDER - Needs Kaggle training
```

### CORE 2: NLPFinBERT
```python
config = {
    "base_model": "ProsusAI/finbert",
    "hidden_dim": 768,
    "num_classes": 3,     # Positive, Negative, Neutral
    "max_length": 512,
    "learning_rate": 2e-5
}
# Status: PLACEHOLDER - Needs Kaggle fine-tuning
```

### CORE 3: MetaDecision
```python
config = {
    "quant_dim": 512,     # From CORE 1
    "nlp_dim": 768,       # From CORE 2
    "hidden_dim": 256,
    "n_actions": 3,       # BUY, SELL, HOLD
    "rl_gamma": 0.99,
    "rl_lr": 1e-4
}
# Status: PLACEHOLDER - Needs training after CORE 1 & 2
```

---

## 📊 CURRENT BENCHMARK

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| XGBoost Accuracy | 49.91% | >50% | ❌ |
| Walk-Forward Sharpe | 1.62 | >1.0 | ✅ |
| Stability Score | 0.0 | >0.5 | ❌ |
| Hold-out Sharpe | -1.188 | >0.0 | ❌ |
| Hold-out Accuracy | 49.9% | >50% | ❌ |
| Phi (Consciousness) | 0.315 | >0.3 | ✅ |

**Verdict:** `ready_for_production: false`

---

## 🔑 KEY IMPORTS

```python
# Main entry point
from src import WalkForwardOptimizer, PhiCalculator, IITCore

# Data pipeline
from src.data.data_pipeline import DataPipeline

# Guards
from src.guards import QuantGuard, NLPGuard

# Strategy
from src.strategy.phi_filter import PhiFilter, create_phi_filter

# Backtesting
from src.backtesting import AdvancedWalkForwardOptimizer, ProductionBacktester

# Risk
from src.risk.position_sizer import PositionSizer
from src.risk.advanced_risk_manager import AdvancedRiskManager
```

---

## 📝 NOTES FOR FUTURE DEVELOPMENT

1. **Consciousness module** was created on Dec 24, 2025 - uses simplified IIT Phi proxies
2. **CORE models** are placeholders - need GPU training on Kaggle
3. **Notebooks 04 & 05** missing - need to create for CORE training
4. **Twitter API** needed for full social loader functionality
5. **Target: Production-ready system with $30K-$60K sale value**
