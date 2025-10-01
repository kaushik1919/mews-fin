"""Hypothesis testing suite for MEWS research workflows."""

from .reporting import HypothesisReportBuilder
from .results import (
    GrangerCausalityResult,
    GrangerLagResult,
    LikelihoodRatioResult,
    PairedTestResult,
)
from .tests import (
    granger_causality,
    likelihood_ratio_test,
    paired_t_test,
    permutation_test,
)

__all__ = [
    "HypothesisReportBuilder",
    "GrangerCausalityResult",
    "GrangerLagResult",
    "LikelihoodRatioResult",
    "PairedTestResult",
    "granger_causality",
    "likelihood_ratio_test",
    "paired_t_test",
    "permutation_test",
]
