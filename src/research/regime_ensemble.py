"""Backward-compatible exports for research modules."""

from __future__ import annotations

from src.ensemble.regime import RegimeAdaptiveEnsemble, VolatilityRegimeDetector

__all__ = [
    "RegimeAdaptiveEnsemble",
    "VolatilityRegimeDetector",
]
