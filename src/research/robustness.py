"""Robustness and ethics diagnostics for MEWS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None  # type: ignore


@dataclass
class SentimentBiasReport:
    group_a: str
    group_b: str
    mean_a: float
    mean_b: float
    ks_statistic: float
    p_value: float
    significant: bool


class SentimentBiasDetector:
    def __init__(self, sentiment_col: str = "sentiment_score") -> None:
        self.sentiment_col = sentiment_col

    def compare_groups(
        self,
        df: pd.DataFrame,
        group_col: str,
        group_a: str,
        group_b: str,
    ) -> SentimentBiasReport:
        if stats is None:
            raise ImportError("scipy is required for sentiment bias detection")

        subset_a = df[df[group_col] == group_a][self.sentiment_col].dropna()
        subset_b = df[df[group_col] == group_b][self.sentiment_col].dropna()

        if subset_a.empty or subset_b.empty:
            raise ValueError("Both groups must contain sentiment observations")

        ks_stat, pvalue = stats.ks_2samp(subset_a, subset_b)
        report = SentimentBiasReport(
            group_a=group_a,
            group_b=group_b,
            mean_a=float(subset_a.mean()),
            mean_b=float(subset_b.mean()),
            ks_statistic=float(ks_stat),
            p_value=float(pvalue),
            significant=bool(pvalue < 0.05),
        )
        return report


class RobustnessStressTester:
    def __init__(
        self, date_col: str = "Date", sentiment_col: str = "sentiment_score"
    ) -> None:
        self.date_col = date_col
        self.sentiment_col = sentiment_col

    def inject_noise(
        self,
        df: pd.DataFrame,
        noise_level: float = 0.1,
        columns: Optional[list[str]] = None,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        working = df.copy()
        numeric_cols = (
            columns or working.select_dtypes(include=[np.number]).columns.tolist()
        )
        for col in numeric_cols:
            col_std = working[col].std()
            if np.isnan(col_std) or col_std == 0:
                continue
            noise = rng.normal(0, col_std * noise_level, size=len(working))
            working[col] = working[col] + noise
        return working

    def simulate_news_delay(
        self,
        df: pd.DataFrame,
        delay_days: int = 1,
        group_cols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        working = df.copy()
        working[self.date_col] = pd.to_datetime(working[self.date_col])
        if group_cols:
            working.sort_values(group_cols + [self.date_col], inplace=True)
            if self.sentiment_col in working.columns:
                shifted = working.groupby(group_cols)[self.sentiment_col].shift(
                    delay_days
                )
                working[self.sentiment_col] = shifted.fillna(method="bfill")
        else:
            working.sort_values(self.date_col, inplace=True)
            for col in working.select_dtypes(include=[np.number]).columns:
                working[col] = working[col].shift(delay_days).fillna(method="bfill")
        return working


__all__ = [
    "SentimentBiasReport",
    "SentimentBiasDetector",
    "RobustnessStressTester",
]
