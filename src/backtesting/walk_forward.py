#!/usr/bin/env python3
"""
PhiHorizon - Walk-Forward Optimization Framework

Advanced walk-forward optimization with Bayesian optimization, out-of-sample validation,
and comprehensive overfitting prevention techniques.

Based on cutting-edge research in algorithmic trading optimization:
- Bayesian optimization for efficient parameter search
- Multiple validation metrics (Sharpe, Sortino, Calmar, max drawdown)
- Statistical significance testing
- Overfitting detection algorithms
- Rolling window validation
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ..strategy.base import BaseStrategy


@dataclass
class OptimizationResult:
    """Container for walk-forward optimization results."""

    window_idx: int
    in_sample_start: datetime
    in_sample_end: datetime
    out_sample_start: datetime
    out_sample_end: datetime
    optimal_params: Dict[str, Any]
    in_sample_metrics: Dict[str, float]
    out_sample_metrics: Dict[str, float]
    validation_score: float
    overfitting_risk: float
    statistical_significance: float


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward optimization."""

    in_sample_periods: int = 252  # ~1 year
    out_sample_periods: int = 63  # ~1 quarter
    step_size: int = 21  # ~1 month step
    min_samples: int = 3  # Minimum walk-forward windows

    # Optimization settings
    optimization_method: str = "bayesian"  # 'bayesian', 'grid', 'random'
    max_evaluations: int = 100
    early_stopping_rounds: int = 10

    # Validation settings
    validation_metrics: List[str] = None
    significance_level: float = 0.05
    min_sharpe_ratio: float = 1.0
    max_drawdown_limit: float = 0.20

    def __post_init__(self) -> None:
        if self.validation_metrics is None:
            self.validation_metrics = [
                "sharpe_ratio",
                "sortino_ratio",
                "calmar_ratio",
                "total_return",
            ]


class AdvancedWalkForwardOptimizer:
    """
    Advanced Walk-Forward Optimization Framework

    Implements state-of-the-art optimization techniques:
    - Bayesian optimization for efficient parameter search
    - Rolling walk-forward validation
    - Statistical significance testing
    - Overfitting detection and prevention
    - Multiple performance metrics validation
    """

    def __init__(self, config: WalkForwardConfig = None) -> None:
        """
        Initialize the walk-forward optimizer.

        Args:
            config: Configuration object for optimization parameters
        """
        self.config = config or WalkForwardConfig()
        self.logger = logging.getLogger(__name__)

        # Bayesian optimization state
        self.gaussian_process = None
        self.optimization_history = []
        self.best_params_history = []

        # Validation results
        self.walk_forward_results = []
        self.overfitting_analysis = {}

    def optimize_strategy(
        self,
        strategy_class: type,
        data: pd.DataFrame,
        param_ranges: Dict[str, Dict[str, Union[int, float]]],
        fixed_params: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive walk-forward optimization.

        Args:
            strategy_class: Strategy class to optimize
            data: Historical OHLCV data
            param_ranges: Parameter ranges for optimization
            fixed_params: Fixed parameters not to optimize

        Returns:
            Dictionary with optimization results and recommendations
        """
        try:
            self.logger.info("Starting advanced walk-forward optimization...")

            # Perform walk-forward analysis
            wf_results = self._perform_walk_forward_analysis(
                strategy_class, data, param_ranges, fixed_params or {}
            )

            # Analyze overfitting risk
            overfitting_analysis = self._analyze_overfitting_risk(wf_results)

            # Statistical validation
            statistical_validation = self._perform_statistical_validation(wf_results)

            # Generate final recommendations
            recommendations = self._generate_optimization_recommendations(
                wf_results, overfitting_analysis, statistical_validation
            )

            results = {
                "walk_forward_results": wf_results,
                "overfitting_analysis": overfitting_analysis,
                "statistical_validation": statistical_validation,
                "recommendations": recommendations,
                "optimization_summary": self._create_optimization_summary(wf_results),
            }

            self.logger.info(
                f"Walk-forward optimization completed. " f"Processed {len(wf_results)} windows."
            )

            return results

        except Exception as e:
            self.logger.error(f"Walk-forward optimization failed: {e}")
            raise

    def _perform_walk_forward_analysis(
        self,
        strategy_class: type,
        data: pd.DataFrame,
        param_ranges: Dict[str, Dict[str, Union[int, float]]],
        fixed_params: Dict[str, Any],
    ) -> List[OptimizationResult]:
        """
        Perform rolling walk-forward analysis.

        Args:
            strategy_class: Strategy class to optimize
            data: Historical data
            param_ranges: Parameter ranges
            fixed_params: Fixed parameters

        Returns:
            List of optimization results for each window
        """
        results = []
        total_periods = len(data)

        # Calculate number of walk-forward windows
        num_windows = max(
            self.config.min_samples,
            (total_periods - self.config.in_sample_periods - self.config.out_sample_periods)
            // self.config.step_size,
        )

        self.logger.info(f"Performing walk-forward analysis with {num_windows} windows...")

        for window_idx in range(num_windows):
            try:
                # Define window boundaries
                start_idx = window_idx * self.config.step_size
                in_sample_end = start_idx + self.config.in_sample_periods
                out_sample_end = in_sample_end + self.config.out_sample_periods

                if out_sample_end > total_periods:
                    break

                # Split data
                in_sample_data = data.iloc[start_idx:in_sample_end].copy()
                out_sample_data = data.iloc[in_sample_end:out_sample_end].copy()

                # Convert timestamps for logging
                in_sample_start_date = (
                    in_sample_data["timestamp"].iloc[0] if len(in_sample_data) > 0 else None
                )
                in_sample_end_date = (
                    in_sample_data["timestamp"].iloc[-1] if len(in_sample_data) > 0 else None
                )
                out_sample_start_date = (
                    out_sample_data["timestamp"].iloc[0] if len(out_sample_data) > 0 else None
                )
                out_sample_end_date = (
                    out_sample_data["timestamp"].iloc[-1] if len(out_sample_data) > 0 else None
                )

                self.logger.info(
                    f"Window {window_idx+1}/{num_windows}: "
                    f"Train [{in_sample_start_date} to {in_sample_end_date}], "
                    f"Test [{out_sample_start_date} to {out_sample_end_date}]"
                )

                # Optimize parameters on in-sample data
                optimal_params = self._optimize_parameters_bayesian(
                    strategy_class, in_sample_data, param_ranges, fixed_params
                )

                # Evaluate on in-sample data
                in_sample_metrics = self._evaluate_strategy(
                    strategy_class, in_sample_data, {**optimal_params, **fixed_params}
                )

                # Evaluate on out-of-sample data
                out_sample_metrics = self._evaluate_strategy(
                    strategy_class, out_sample_data, {**optimal_params, **fixed_params}
                )

                # Calculate validation score and overfitting risk
                validation_score = self._calculate_validation_score(
                    in_sample_metrics, out_sample_metrics
                )

                overfitting_risk = self._calculate_overfitting_risk(
                    in_sample_metrics, out_sample_metrics
                )

                # Statistical significance test
                statistical_significance = self._test_statistical_significance(out_sample_metrics)

                # Store results
                result = OptimizationResult(
                    window_idx=window_idx + 1,
                    in_sample_start=in_sample_start_date,
                    in_sample_end=in_sample_end_date,
                    out_sample_start=out_sample_start_date,
                    out_sample_end=out_sample_end_date,
                    optimal_params=optimal_params,
                    in_sample_metrics=in_sample_metrics,
                    out_sample_metrics=out_sample_metrics,
                    validation_score=validation_score,
                    overfitting_risk=overfitting_risk,
                    statistical_significance=statistical_significance,
                )

                results.append(result)

            except Exception as e:
                self.logger.warning(f"Failed to process window {window_idx+1}: {e}")
                continue

        return results

    def _optimize_parameters_bayesian(
        self,
        strategy_class: type,
        data: pd.DataFrame,
        param_ranges: Dict[str, Dict[str, Union[int, float]]],
        fixed_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Bayesian optimization for parameter tuning.

        Uses Gaussian Process optimization for efficient parameter search.
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Integer, Real
            from skopt.utils import use_named_args

            # Define search space
            dimensions = []
            param_names = []

            for param_name, param_range in param_ranges.items():
                param_names.append(param_name)

                if "type" in param_range:
                    param_type = param_range["type"]
                elif isinstance(param_range.get("min", 0), int) and isinstance(
                    param_range.get("max", 1), int
                ):
                    param_type = "int"
                else:
                    param_type = "float"

                if param_type == "int":
                    dimensions.append(
                        Integer(param_range["min"], param_range["max"], name=param_name)
                    )
                else:
                    dimensions.append(
                        Real(
                            param_range["min"], param_range["max"], prior="uniform", name=param_name
                        )
                    )

            # Objective function for Bayesian optimization
            @use_named_args(dimensions)
            def objective(**params) -> float:
                try:
                    # Combine with fixed parameters
                    all_params = {**params, **fixed_params}

                    # Evaluate strategy
                    metrics = self._evaluate_strategy(strategy_class, data, all_params)

                    # Use negative Sharpe ratio (minimize negative = maximize Sharpe)
                    return -metrics.get("sharpe_ratio", -999)

                except Exception as e:
                    self.logger.warning(f"Parameter evaluation failed: {e}")
                    return 999  # Return high penalty

            # Perform Bayesian optimization
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = gp_minimize(
                    objective,
                    dimensions,
                    n_calls=self.config.max_evaluations,
                    n_random_starts=max(10, len(dimensions) * 2),
                    random_state=42,
                )

            # Extract optimal parameters
            optimal_params = {}
            for i, param_name in enumerate(param_names):
                optimal_params[param_name] = result.x[i]

            return optimal_params

        except ImportError:
            self.logger.warning("scikit-optimize not available, falling back to grid search")
            return self._optimize_parameters_grid(strategy_class, data, param_ranges, fixed_params)

    def _optimize_parameters_grid(
        self,
        strategy_class: type,
        data: pd.DataFrame,
        param_ranges: Dict[str, Dict[str, Union[int, float]]],
        fixed_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Grid search optimization as fallback.
        """
        # Generate parameter combinations (simplified version)
        param_combinations = self._generate_param_combinations(param_ranges)

        best_params = None
        best_score = -999

        for params in param_combinations[: min(50, len(param_combinations))]:  # Limit combinations
            try:
                all_params = {**params, **fixed_params}
                metrics = self._evaluate_strategy(strategy_class, data, all_params)
                score = metrics.get("sharpe_ratio", -999)

                if score > best_score:
                    best_score = score
                    best_params = params

            except Exception as e:
                continue

        return best_params or {}

    def _generate_param_combinations(
        self, param_ranges: Dict[str, Dict[str, Union[int, float]]]
    ) -> List[Dict[str, Any]]:
        """
        Generate parameter combinations for grid search.
        """
        import itertools

        # Simplified: generate 3 values per parameter
        combinations = []

        param_values = {}
        for param_name, param_range in param_ranges.items():
            min_val = param_range["min"]
            max_val = param_range["max"]

            if isinstance(min_val, int) and isinstance(max_val, int):
                # Integer parameter
                step = max(1, (max_val - min_val) // 3)
                values = list(range(min_val, max_val + 1, step))
            else:
                # Float parameter
                step = (max_val - min_val) / 3
                values = [min_val + i * step for i in range(4)]

            param_values[param_name] = values

        # Generate all combinations
        param_names = list(param_values.keys())
        for combo in itertools.product(*[param_values[name] for name in param_names]):
            combinations.append(dict(zip(param_names, combo)))

        return combinations

    def _evaluate_strategy(
        self, strategy_class: type, data: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Evaluate strategy performance with given parameters.
        """
        try:
            # Create strategy instance
            strategy = strategy_class(**params)

            # Run backtest (import here to avoid circular import)
            from ..backtesting.production_backtester import ProductionBacktester

            backtester = ProductionBacktester()
            results = backtester.run_backtest(strategy, data)

            # Extract key metrics
            metrics = {
                "total_return": results.get("total_return", 0),
                "sharpe_ratio": results.get("sharpe_ratio", 0),
                "sortino_ratio": results.get("sortino_ratio", 0),
                "calmar_ratio": results.get("calmar_ratio", 0),
                "max_drawdown": results.get("max_drawdown", 0),
                "win_rate": results.get("win_rate", 0),
                "profit_factor": results.get("profit_factor", 1),
                "total_trades": results.get("total_trades", 0),
            }

            return metrics

        except Exception as e:
            self.logger.warning(f"Strategy evaluation failed: {e}")
            return {
                "total_return": 0,
                "sharpe_ratio": -999,
                "sortino_ratio": -999,
                "calmar_ratio": -999,
                "max_drawdown": 1,
                "win_rate": 0,
                "profit_factor": 0,
                "total_trades": 0,
            }

    def _calculate_validation_score(
        self, in_sample_metrics: Dict[str, float], out_sample_metrics: Dict[str, float]
    ) -> float:
        """
        Calculate validation score based on in-sample vs out-sample performance.
        """
        try:
            # Primary metric: Sharpe ratio stability
            in_sharpe = in_sample_metrics.get("sharpe_ratio", 0)
            out_sharpe = out_sample_metrics.get("sharpe_ratio", 0)

            # Sharpe ratio retention (closer to 1.0 is better)
            sharpe_retention = min(out_sharpe / in_sharpe, 1.0) if in_sharpe > 0 else 0

            # Return stability
            in_return = in_sample_metrics.get("total_return", 0)
            out_return = out_sample_metrics.get("total_return", 0)
            return_stability = 1 - abs(in_return - out_return) / max(abs(in_return), 0.01)

            # Drawdown control
            out_drawdown = out_sample_metrics.get("max_drawdown", 1)
            drawdown_penalty = max(0, out_drawdown - self.config.max_drawdown_limit)

            # Combined score
            validation_score = (
                sharpe_retention * 0.5 + return_stability * 0.3 + (1 - drawdown_penalty) * 0.2
            )

            return max(0, min(1, validation_score))

        except Exception as e:
            self.logger.warning(f"Validation score calculation failed: {e}")
            return 0.5

    def _calculate_overfitting_risk(
        self, in_sample_metrics: Dict[str, float], out_sample_metrics: Dict[str, float]
    ) -> float:
        """
        Calculate overfitting risk based on performance degradation.
        """
        try:
            # Multiple overfitting indicators
            overfitting_indicators = []

            # Sharpe ratio degradation
            in_sharpe = in_sample_metrics.get("sharpe_ratio", 0)
            out_sharpe = out_sample_metrics.get("sharpe_ratio", 0)
            if in_sharpe > 0:
                sharpe_degradation = max(0, (in_sharpe - out_sharpe) / in_sharpe)
                overfitting_indicators.append(sharpe_degradation)

            # Return degradation
            in_return = abs(in_sample_metrics.get("total_return", 0))
            out_return = abs(out_sample_metrics.get("total_return", 0))
            if in_return > 0:
                return_degradation = max(0, (in_return - out_return) / in_return)
                overfitting_indicators.append(return_degradation)

            # Win rate consistency
            in_win_rate = in_sample_metrics.get("win_rate", 0)
            out_win_rate = out_sample_metrics.get("win_rate", 0)
            win_rate_consistency = abs(in_win_rate - out_win_rate)
            overfitting_indicators.append(win_rate_consistency)

            # Average overfitting risk
            if overfitting_indicators:
                avg_overfitting = np.mean(overfitting_indicators)
                return min(1.0, avg_overfitting)
            else:
                return 0.5

        except Exception as e:
            self.logger.warning(f"Overfitting risk calculation failed: {e}")
            return 0.5

    def _test_statistical_significance(self, out_sample_metrics: Dict[str, float]) -> float:
        """
        Test statistical significance of out-sample performance.
        """
        try:
            # Test if Sharpe ratio is significantly different from zero
            # This is a simplified test - in practice would use more sophisticated methods
            sharpe_ratio = out_sample_metrics.get("sharpe_ratio", 0)
            total_return = out_sample_metrics.get("total_return", 0)
            max_drawdown = out_sample_metrics.get("max_drawdown", 0)

            # Simple significance test: Sharpe > 1 and positive return with controlled drawdown
            significance_score = 0

            if sharpe_ratio > 1.0:
                significance_score += 0.4
            elif sharpe_ratio > 0.5:
                significance_score += 0.2

            if total_return > 0:
                significance_score += 0.3

            if max_drawdown < 0.15:
                significance_score += 0.3

            return min(1.0, significance_score)

        except Exception as e:
            self.logger.warning(f"Statistical significance test failed: {e}")
            return 0.0

    def _analyze_overfitting_risk(self, wf_results: List[OptimizationResult]) -> Dict[str, Any]:
        """
        Comprehensive overfitting analysis across all walk-forward windows.
        """
        try:
            if not wf_results:
                return {"error": "No walk-forward results available"}

            # Extract metrics
            validation_scores = [r.validation_score for r in wf_results]
            overfitting_risks = [r.overfitting_risk for r in wf_results]
            sharpe_ratios = [r.out_sample_metrics.get("sharpe_ratio", 0) for r in wf_results]

            # Statistical analysis
            analysis = {
                "average_validation_score": np.mean(validation_scores),
                "validation_score_std": np.std(validation_scores),
                "average_overfitting_risk": np.mean(overfitting_risks),
                "overfitting_risk_std": np.std(overfitting_risks),
                "sharpe_ratio_persistence": self._calculate_persistence(sharpe_ratios),
                "validation_score_trend": self._calculate_trend(validation_scores),
                "overfitting_risk_trend": self._calculate_trend(overfitting_risks),
            }

            # Risk assessment
            if analysis["average_overfitting_risk"] > 0.7:
                analysis["overfitting_assessment"] = "HIGH RISK"
            elif analysis["average_overfitting_risk"] > 0.5:
                analysis["overfitting_assessment"] = "MEDIUM RISK"
            else:
                analysis["overfitting_assessment"] = "LOW RISK"

            # Validation score assessment
            if analysis["average_validation_score"] > 0.8:
                analysis["validation_assessment"] = "EXCELLENT"
            elif analysis["average_validation_score"] > 0.6:
                analysis["validation_assessment"] = "GOOD"
            else:
                analysis["validation_assessment"] = "POOR"

            return analysis

        except Exception as e:
            self.logger.warning(f"Overfitting analysis failed: {e}")
            return {"error": str(e)}

    def _calculate_persistence(self, values: List[float]) -> float:
        """Calculate persistence of performance across windows."""
        if len(values) < 2:
            return 0.5

        # Count consistent positive performance
        positive_count = sum(1 for v in values if v > 0)
        persistence = positive_count / len(values)

        return persistence

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values across windows."""
        if len(values) < 3:
            return 0.0

        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        # Normalize to [-1, 1] range
        return max(-1, min(1, slope * len(values)))

    def _perform_statistical_validation(
        self, wf_results: List[OptimizationResult]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive statistical validation of results.
        """
        try:
            if not wf_results:
                return {"error": "No walk-forward results available"}

            # Extract metrics for statistical tests
            sharpe_ratios = [r.out_sample_metrics.get("sharpe_ratio", 0) for r in wf_results]
            total_returns = [r.out_sample_metrics.get("total_return", 0) for r in wf_results]

            # Basic statistical tests
            validation = {
                "sharpe_ratio_mean": np.mean(sharpe_ratios),
                "sharpe_ratio_std": np.std(sharpe_ratios),
                "sharpe_ratio_skewness": stats.skew(sharpe_ratios) if len(sharpe_ratios) > 2 else 0,
                "return_mean": np.mean(total_returns),
                "return_std": np.std(total_returns),
                "sharpe_positive_ratio": sum(1 for s in sharpe_ratios if s > 0)
                / len(sharpe_ratios),
                "return_positive_ratio": sum(1 for r in total_returns if r > 0)
                / len(total_returns),
            }

            # T-test for Sharpe ratio significance
            try:
                t_stat, p_value = stats.ttest_1samp(sharpe_ratios, 0)
                validation["sharpe_t_stat"] = t_stat
                validation["sharpe_p_value"] = p_value
                validation["sharpe_significant"] = p_value < self.config.significance_level
            except (ValueError, TypeError, ZeroDivisionError):
                validation["sharpe_t_stat"] = None
                validation["sharpe_p_value"] = None
                validation["sharpe_significant"] = False

            # Performance assessment
            if validation["sharpe_ratio_mean"] > 1.0 and validation["sharpe_positive_ratio"] > 0.7:
                validation["performance_rating"] = "EXCELLENT"
            elif (
                validation["sharpe_ratio_mean"] > 0.5 and validation["sharpe_positive_ratio"] > 0.6
            ):
                validation["performance_rating"] = "GOOD"
            elif validation["sharpe_ratio_mean"] > 0 and validation["sharpe_positive_ratio"] > 0.5:
                validation["performance_rating"] = "FAIR"
            else:
                validation["performance_rating"] = "POOR"

            return validation

        except Exception as e:
            self.logger.warning(f"Statistical validation failed: {e}")
            return {"error": str(e)}

    def _generate_optimization_recommendations(
        self,
        wf_results: List[OptimizationResult],
        overfitting_analysis: Dict[str, Any],
        statistical_validation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate actionable recommendations based on analysis.
        """
        try:
            recommendations = {
                "use_strategy": False,
                "confidence_level": "LOW",
                "recommended_parameters": {},
                "risk_warnings": [],
                "optimization_suggestions": [],
            }

            # Decision logic
            avg_validation_score = overfitting_analysis.get("average_validation_score", 0)
            avg_overfitting_risk = overfitting_analysis.get("average_overfitting_risk", 1)
            performance_rating = statistical_validation.get("performance_rating", "POOR")

            # Primary decision criteria
            if (
                avg_validation_score > 0.7
                and avg_overfitting_risk < 0.5
                and performance_rating in ["EXCELLENT", "GOOD"]
            ):

                recommendations["use_strategy"] = True
                recommendations["confidence_level"] = "HIGH"

                # Find most stable parameters across windows
                recommendations["recommended_parameters"] = self._find_stable_parameters(wf_results)

            elif (
                avg_validation_score > 0.5
                and avg_overfitting_risk < 0.7
                and performance_rating in ["GOOD", "FAIR"]
            ):

                recommendations["use_strategy"] = True
                recommendations["confidence_level"] = "MEDIUM"
                recommendations["recommended_parameters"] = self._find_stable_parameters(wf_results)

            else:
                recommendations["use_strategy"] = False
                recommendations["confidence_level"] = "LOW"

            # Generate risk warnings
            if avg_overfitting_risk > 0.7:
                recommendations["risk_warnings"].append(
                    "High overfitting risk detected. Parameters may not generalize well."
                )

            if avg_validation_score < 0.5:
                recommendations["risk_warnings"].append(
                    "Poor out-of-sample validation. Strategy may be curve-fitted."
                )

            if statistical_validation.get("sharpe_positive_ratio", 0) < 0.5:
                recommendations["risk_warnings"].append(
                    "Inconsistent performance across time periods."
                )

            # Optimization suggestions
            if len(wf_results) < self.config.min_samples:
                recommendations["optimization_suggestions"].append(
                    f"Increase walk-forward windows (current: {len(wf_results)}, recommended: {self.config.min_samples}+)"
                )

            if overfitting_analysis.get("validation_score_trend", 0) < -0.5:
                recommendations["optimization_suggestions"].append(
                    "Validation scores declining over time. Consider shorter optimization periods."
                )

            return recommendations

        except Exception as e:
            self.logger.warning(f"Recommendation generation failed: {e}")
            return {"error": str(e)}

    def _find_stable_parameters(self, wf_results: List[OptimizationResult]) -> Dict[str, Any]:
        """
        Find the most stable parameters across walk-forward windows.
        """
        try:
            if not wf_results:
                return {}

            # Collect all parameter sets
            param_sets = [r.optimal_params for r in wf_results]
            validation_scores = [r.validation_score for r in wf_results]

            # Weight by validation score
            if len(param_sets) > 1:
                # Find parameters that appear most frequently with high validation scores
                stable_params = {}

                # Get all parameter names
                param_names = set()
                for param_set in param_sets:
                    param_names.update(param_set.keys())

                # For each parameter, find the most common value among high-validation results
                for param_name in param_names:
                    # Get values with validation score > median
                    median_score = np.median(validation_scores)
                    good_values = [
                        param_sets[i][param_name]
                        for i in range(len(param_sets))
                        if validation_scores[i] >= median_score and param_name in param_sets[i]
                    ]

                    if good_values:
                        # Use most common value, or mean if numeric
                        try:
                            stable_params[param_name] = np.mean(good_values)
                        except (TypeError, ValueError):
                            # For non-numeric, use most common
                            stable_params[param_name] = max(set(good_values), key=good_values.count)

                return stable_params
            else:
                return param_sets[0] if param_sets else {}

        except Exception as e:
            self.logger.warning(f"Stable parameter detection failed: {e}")
            return {}

    def _create_optimization_summary(self, wf_results: List[OptimizationResult]) -> Dict[str, Any]:
        """
        Create comprehensive optimization summary.
        """
        try:
            summary = {
                "total_windows": len(wf_results),
                "successful_windows": sum(1 for r in wf_results if r.validation_score > 0.5),
                "average_validation_score": np.mean([r.validation_score for r in wf_results]),
                "average_overfitting_risk": np.mean([r.overfitting_risk for r in wf_results]),
                "best_window_idx": (
                    max(wf_results, key=lambda r: r.validation_score).window_idx
                    if wf_results
                    else None
                ),
                "worst_window_idx": (
                    min(wf_results, key=lambda r: r.validation_score).window_idx
                    if wf_results
                    else None
                ),
                "optimization_timestamp": datetime.now().isoformat(),
            }

            return summary

        except Exception as e:
            self.logger.warning(f"Optimization summary creation failed: {e}")
            return {"error": str(e)}


# Convenience function for easy optimization
def optimize_strategy_walk_forward(
    strategy_class: type,
    data: pd.DataFrame,
    param_ranges: Dict[str, Dict[str, Union[int, float]]],
    config: WalkForwardConfig = None,
) -> Dict[str, Any]:
    """
    Convenience function for walk-forward optimization.

    Args:
        strategy_class: Strategy class to optimize
        data: Historical data
        param_ranges: Parameter ranges for optimization
        config: Walk-forward configuration

    Returns:
        Optimization results dictionary
    """
    optimizer = AdvancedWalkForwardOptimizer(config)
    return optimizer.optimize_strategy(strategy_class, data, param_ranges)

