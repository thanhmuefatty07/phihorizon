#!/usr/bin/env python3
"""
PhiHorizon V7.0 Core Package

3-Core Deep Learning Architecture:
- CORE 1: Quant Transformer (numerical data)
- CORE 2: NLP FinBERT (text/sentiment)
- CORE 3: Meta Decision Engine (fusion + decision)
"""

from .quant_transformer import QuantTransformer, QuantTransformerConfig
from .nlp_finbert import NLPFinBERT, NLPFinBERTConfig
from .meta_decision import MetaDecisionEngine, MetaDecisionConfig

__all__ = [
    "QuantTransformer",
    "QuantTransformerConfig",
    "NLPFinBERT",
    "NLPFinBERTConfig",
    "MetaDecisionEngine",
    "MetaDecisionConfig"
]
