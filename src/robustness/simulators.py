"""Simulators for introducing delays or missing data into news features."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class DelayReport:
    delay_days: int
    missing_ratio: float
    affected_columns: Tuple[str, ...]
    group_columns: Tuple[str, ...]
    random_state: Optional[int]


class DelaySimulator:
    """Shift and drop news-derived features to emulate delayed availability."""

    def __init__(
        self,
        date_col: str = "Date",
        sentinel_substrings: Sequence[str] = ("sentiment", "news", "headline"),
    ) -> None:
        self.date_col = date_col
        self.sentinel_substrings = tuple(s.lower() for s in sentinel_substrings)
        self.logger = LOGGER

    def apply(
        self,
        df: pd.DataFrame,
        delay_days: int = 1,
        target_columns: Optional[Sequence[str]] = None,
        group_columns: Optional[Sequence[str]] = None,
        missing_ratio: float = 0.0,
        random_state: Optional[int] = None,
    ) -> tuple[pd.DataFrame, DelayReport]:
        """Return perturbed dataframe and metadata about the simulated delay."""

        if delay_days < 0:
            raise ValueError("delay_days must be non-negative")
        if not 0.0 <= missing_ratio <= 1.0:
            raise ValueError("missing_ratio must be between 0 and 1")

        working = df.copy()
        if self.date_col not in working.columns:
            raise ValueError(f"Dataframe must contain date column '{self.date_col}'")

        working[self.date_col] = pd.to_datetime(working[self.date_col])

        if target_columns is None:
            target_columns = [
                col
                for col in working.columns
                if any(token in col.lower() for token in self.sentinel_substrings)
            ]
        else:
            target_columns = [col for col in target_columns if col in working.columns]

        affected_columns = tuple(target_columns)
        group_columns = tuple(group_columns or ())

        if delay_days > 0 and affected_columns:
            if group_columns:
                working.sort_values(list(group_columns) + [self.date_col], inplace=True)
                grouped = working.groupby(list(group_columns), sort=False)
                for column in affected_columns:
                    working[column] = (
                        grouped[column]
                        .shift(delay_days)
                        .fillna(method="bfill")
                    )
            else:
                working.sort_values(self.date_col, inplace=True)
                for column in affected_columns:
                    working[column] = working[column].shift(delay_days).fillna(method="bfill")

        if missing_ratio > 0 and affected_columns:
            rng = np.random.default_rng(random_state)
            mask = rng.random(len(working)) < missing_ratio
            if mask.any():
                for column in affected_columns:
                    working.loc[mask, column] = np.nan

        # Restore original row ordering so downstream joins remain stable
        working.sort_index(kind="stable", inplace=True)

        report = DelayReport(
            delay_days=int(delay_days),
            missing_ratio=float(missing_ratio),
            affected_columns=affected_columns,
            group_columns=group_columns,
            random_state=random_state,
        )
        return working, report


__all__ = ["DelaySimulator", "DelayReport"]
