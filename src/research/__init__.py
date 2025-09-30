"""Research-focused extensions for the Market Risk Early Warning System."""

from .regime_ensemble import RegimeAdaptiveEnsemble, VolatilityRegimeDetector
from .cross_modal import CrossAttentionFusion
from .evaluation import (
    BenchmarkSuite,
    CrisisPeriod,
    ResearchEvaluator,
    baseline_garch_var,
    baseline_lstm,
)
from .hypothesis import (
    SentimentImpactTester,
    GraphFeatureAblation,
)
from .robustness import (
    SentimentBiasReport,
    RobustnessStressTester,
)
from .reporting import ResearchReportBuilder

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
