"""Adversarial noise utilities for robustness testing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class NoiseReport:
    noise_level: float
    affected_columns: Sequence[str]
    random_state: Optional[int]


class AdversarialNoiseTester:
    """Injects perturbations into tabular features to test model resilience."""

    def __init__(self, numeric_only: bool = True) -> None:
        self.numeric_only = numeric_only
        self.logger = LOGGER

    def apply(
        self,
        df: pd.DataFrame,
        noise_level: float = 0.05,
        columns: Optional[Sequence[str]] = None,
        random_state: Optional[int] = None,
        strategy: str = "gaussian",
    ) -> tuple[pd.DataFrame, NoiseReport]:
        """Return a perturbed copy of ``df`` and metadata about the noise injection."""

        if noise_level < 0:
            raise ValueError("noise_level must be non-negative")

        working = df.copy()
        rng = np.random.default_rng(random_state)

        if columns is None:
            if self.numeric_only:
                candidate_columns = working.select_dtypes(
                    include=[np.number]
                ).columns.tolist()
            else:
                candidate_columns = [
                    col
                    for col in working.columns
                    if pd.api.types.is_numeric_dtype(working[col])
                ]
        else:
            candidate_columns = [col for col in columns if col in working.columns]

        if not candidate_columns:
            self.logger.warning("No columns available for adversarial noise injection")
            return working, NoiseReport(
                noise_level=noise_level, affected_columns=(), random_state=random_state
            )

        for column in candidate_columns:
            series = working[column].astype(float)
            std = float(series.std(ddof=0))
            if not np.isfinite(std) or std == 0:
                continue

            if strategy == "gaussian":
                noise = rng.normal(loc=0.0, scale=std * noise_level, size=len(series))
            elif strategy == "uniform":
                noise = rng.uniform(
                    low=-std * noise_level, high=std * noise_level, size=len(series)
                )
            else:
                raise ValueError(f"Unknown noise strategy: {strategy}")

            working[column] = series + noise

        report = NoiseReport(
            noise_level=float(noise_level),
            affected_columns=tuple(candidate_columns),
            random_state=random_state,
        )
        return working, report


__all__ = ["AdversarialNoiseTester", "NoiseReport"]
