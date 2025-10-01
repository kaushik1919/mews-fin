"""Dataclasses capturing hypothesis test outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PairedTestResult:
    """Result of a paired comparison between two model variants."""

    test_name: str
    metric_name: str
    statistic: float
    p_value: float
    reject_null: bool
    alpha: float
    effect_size: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GrangerLagResult:
    """Summary of a single lag in a Granger causality test."""

    lag: int
    f_statistic: float
    f_pvalue: float
    chi2_statistic: Optional[float]
    chi2_pvalue: Optional[float]


@dataclass
class GrangerCausalityResult:
    """Aggregated Granger causality information across lags."""

    direction: str
    max_lag: int
    results: List[GrangerLagResult] = field(default_factory=list)


@dataclass
class LikelihoodRatioResult:
    """Result of a likelihood ratio test between nested models."""

    model_name: str
    null_loglike: float
    alt_loglike: float
    lr_statistic: float
    degrees_freedom: int
    p_value: float
    reject_null: bool
    alpha: float
    details: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "PairedTestResult",
    "GrangerLagResult",
    "GrangerCausalityResult",
    "LikelihoodRatioResult",
]
