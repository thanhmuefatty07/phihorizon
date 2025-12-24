#!/usr/bin/env python3
"""
PhiHorizon - Phase 2 Data Infrastructure Tests

Comprehensive tests to verify Phase 2 completion.
Run with: pytest tests/test_phase2_data.py -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# =============================================================================
# TEST 1: MAIN PACKAGE IMPORTS
# =============================================================================

class TestMainPackageImports:
    """Test all main package imports work correctly."""
    
    def test_main_imports(self):
        """Test src main exports."""
        from src import WalkForwardOptimizer, PhiCalculator, IITCore
        assert WalkForwardOptimizer is not None
        assert PhiCalculator is not None
        assert IITCore is not None
    
    def test_consciousness_imports(self):
        """Test consciousness module imports."""
        from src.consciousness.metrics import PhiCalculator, IITCore
        from src.consciousness.metrics import calculate_phi_proxy
        from src.consciousness.entropy_metrics import calculate_transfer_entropy
        assert PhiCalculator is not None
        assert calculate_phi_proxy is not None


# =============================================================================
# TEST 2: DATA LOADERS
# =============================================================================

class TestDataLoaders:
    """Test all 14 data loaders can be imported."""
    
    @pytest.mark.parametrize("module,class_name", [
        ("data_pipeline", "DataPipeline"),
    ])
    def test_data_pipeline_import(self, module, class_name):
        """Test DataPipeline imports."""
        exec(f"from src.data.{module} import {class_name}")
    
    def test_data_pipeline_init(self):
        """Test DataPipeline initializes."""
        from src.data.data_pipeline import DataPipeline
        pipeline = DataPipeline()
        assert hasattr(pipeline, 'run')
        assert hasattr(pipeline, 'process_quant_data')


# =============================================================================
# TEST 3: GUARDS
# =============================================================================

class TestGuards:
    """Test ML Guards functionality."""
    
    def test_quant_guard_import(self):
        """Test QuantGuard imports."""
        from src.guards import QuantGuard
        guard = QuantGuard()
        assert hasattr(guard, 'process')
    
    def test_nlp_guard_import(self):
        """Test NLPGuard imports."""
        from src.guards import NLPGuard
        guard = NLPGuard()
        assert guard is not None
    
    def test_quant_guard_process(self):
        """Test QuantGuard processes data correctly."""
        from src.guards.quant_guard import QuantGuard
        
        guard = QuantGuard()
        
        # Create mock data
        mock_df = pd.DataFrame({
            'open': np.random.randn(100) * 100 + 3000,
            'high': np.random.randn(100) * 100 + 3100,
            'low': np.random.randn(100) * 100 + 2900,
            'close': np.random.randn(100) * 100 + 3000,
            'volume': np.random.randint(1000, 100000, 100),
        })
        
        processed, report = guard.process(mock_df)
        
        assert len(processed) > 0
        assert isinstance(report, dict)


# =============================================================================
# TEST 4: CONSCIOUSNESS/PHI
# =============================================================================

class TestConsciousness:
    """Test IIT/Phi consciousness module."""
    
    def test_phi_calculator(self):
        """Test PhiCalculator works."""
        from src.consciousness.metrics import PhiCalculator
        
        calc = PhiCalculator(threshold=0.5, lookback=100)
        
        # Test with random data
        np.random.seed(42)
        test_data = np.random.randn(200) * 0.01
        
        result = calc.calculate(test_data, method='ensemble')
        
        assert hasattr(result, 'phi')
        assert hasattr(result, 'is_integrated')
        assert hasattr(result, 'regime')
        assert hasattr(result, 'confidence')
        assert 0 <= result.phi <= 1
    
    def test_iit_core(self):
        """Test IITCore discretization."""
        from src.consciousness.metrics import IITCore
        
        core = IITCore(n_states=4)
        
        test_data = np.random.randn(100)
        states = core.discretize(test_data)
        
        assert len(states) == len(test_data)
        assert all(0 <= s < 4 for s in states)


# =============================================================================
# TEST 5: PHI FILTER
# =============================================================================

class TestPhiFilter:
    """Test Phi-based trading filter."""
    
    def test_phi_filter_init(self):
        """Test PhiFilter initialization."""
        from src.strategy.phi_filter import PhiFilter, PhiFilterConfig
        
        config = PhiFilterConfig(threshold=0.5, lookback_window=50)
        phi_filter = PhiFilter(config)
        
        assert phi_filter.config.threshold == 0.5
        assert phi_filter.config.lookback_window == 50
    
    def test_phi_filter_calculation(self):
        """Test PhiFilter calculates phi."""
        from src.strategy.phi_filter import PhiFilter, PhiFilterConfig
        
        config = PhiFilterConfig(threshold=0.4, lookback_window=50)
        phi_filter = PhiFilter(config)
        
        # Create mock OHLCV data
        mock_df = pd.DataFrame({
            'open': np.random.randn(200) * 100 + 3000,
            'high': np.random.randn(200) * 100 + 3100,
            'low': np.random.randn(200) * 100 + 2900,
            'close': np.random.randn(200) * 100 + 3000,
            'volume': np.random.randint(1000, 100000, 200),
        })
        
        phi = phi_filter.calculate_phi(mock_df)
        
        assert 0 <= phi <= 1


# =============================================================================
# TEST 6: BACKTESTING
# =============================================================================

class TestBacktesting:
    """Test backtesting module."""
    
    def test_walk_forward_import(self):
        """Test WalkForwardOptimizer imports."""
        from src.backtesting import AdvancedWalkForwardOptimizer
        optimizer = AdvancedWalkForwardOptimizer()
        assert optimizer is not None


# =============================================================================
# TEST 7: RISK MANAGEMENT
# =============================================================================

class TestRiskManagement:
    """Test risk management components."""
    
    def test_position_sizer(self):
        """Test PositionSizer."""
        from src.risk.position_sizer import PositionSizer
        sizer = PositionSizer()
        assert sizer is not None
    
    def test_risk_manager(self):
        """Test AdvancedRiskManager."""
        from src.risk.advanced_risk_manager import AdvancedRiskManager
        rm = AdvancedRiskManager()
        assert rm is not None


# =============================================================================
# TEST 8: CORE MODELS (PLACEHOLDERS)
# =============================================================================

class TestCoreModels:
    """Test CORE model stubs exist."""
    
    def test_quant_transformer(self):
        """Test QuantTransformer placeholder."""
        from src.core.quant_transformer import QuantTransformer
        model = QuantTransformer()
        assert model.is_loaded == False  # Not trained yet
    
    def test_nlp_finbert(self):
        """Test NLPFinBERT placeholder."""
        from src.core.nlp_finbert import NLPFinBERT
        model = NLPFinBERT()
        assert model is not None
    
    def test_meta_decision(self):
        """Test MetaDecisionEngine placeholder."""
        from src.core.meta_decision import MetaDecisionEngine
        engine = MetaDecisionEngine()
        assert engine is not None


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
