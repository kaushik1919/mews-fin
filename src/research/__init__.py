"""Research-focused extensions for the Market Risk Early Warning System."""

from .cross_modal import CrossAttentionFusion
from .evaluation import (
    BenchmarkSuite,
    CrisisPeriod,
    ResearchEvaluator,
    baseline_garch_var,
    baseline_lstm,
)
from .hypothesis import GraphFeatureAblation, SentimentImpactTester
from .regime_ensemble import RegimeAdaptiveEnsemble, VolatilityRegimeDetector
from .reporting import ResearchReportBuilder
from .robustness import RobustnessStressTester, SentimentBiasReport

__all__ = [
    "RegimeAdaptiveEnsemble",
    "VolatilityRegimeDetector",
    "CrossAttentionFusion",
    "BenchmarkSuite",
    "CrisisPeriod",
    "ResearchEvaluator",
    "baseline_garch_var",
    "baseline_lstm",
    "SentimentImpactTester",
    "GraphFeatureAblation",
    "SentimentBiasReport",
    "RobustnessStressTester",
    "ResearchReportBuilder",
]
