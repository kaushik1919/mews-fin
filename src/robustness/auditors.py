"""Sentiment bias diagnostics for MEWS robustness checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

try:  # Optional scientific test dependency
    from scipy import stats
except ImportError:  # pragma: no cover - SciPy is optional in CI
    stats = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class SourceStatistics:
    """Summary statistics for a sentiment signal."""

    column: str
    mean: float
    std: float
    skew: float
    positive_rate: float
    negative_rate: float
    observation_count: int


@dataclass
class BiasSummary:
    source: str
    statistics: List[SourceStatistics] = field(default_factory=list)
    aggregated_mean: float = 0.0
    aggregated_skew: float = 0.0


@dataclass
class BiasComparison:
    reference_source: str
    comparison_source: str
    mean_difference: float
    ks_statistic: Optional[float] = None
    p_value: Optional[float] = None
    significant: Optional[bool] = None


class SentimentBiasAuditor:
    """Evaluate polarity skew between FinBERT and VADER sentiment channels."""

    def __init__(
        self,
        finbert_columns: Optional[Sequence[str]] = None,
        vader_columns: Optional[Sequence[str]] = None,
        positive_threshold: float = 0.05,
        negative_threshold: float = -0.05,
    ) -> None:
        if not finbert_columns and not vader_columns:
            raise ValueError("At least one FinBERT or VADER column must be provided")

        self.finbert_columns = list(finbert_columns or [])
        self.vader_columns = list(vader_columns or [])
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.logger = LOGGER

    def audit(
        self,
        df: pd.DataFrame,
        group_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute bias diagnostics for the configured sentiment columns.

        Args:
            df: DataFrame containing sentiment columns.
            group_col: Optional column used to compute group-level summaries
                (e.g. by symbol, sector, or data source).

        Returns:
            Dictionary with per-source statistics and cross-source comparisons.
        """

        if df.empty:
            raise ValueError("Input dataframe is empty; cannot audit sentiment bias")

        source_payloads: Dict[str, BiasSummary] = {}
        finbert_values: List[float] = []
        vader_values: List[float] = []

        for source, columns in {
            "finbert": self.finbert_columns,
            "vader": self.vader_columns,
        }.items():
            if not columns:
                continue

            statistics: List[SourceStatistics] = []
            column_values: List[np.ndarray] = []
            for column in columns:
                if column not in df.columns:
                    self.logger.debug("Column %s not present in dataframe", column)
                    continue

                series = (
                    df[column].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
                )
                if series.empty:
                    continue

                values = series.to_numpy(dtype=float)
                stats_entry = SourceStatistics(
                    column=column,
                    mean=float(series.mean()),
                    std=float(series.std(ddof=0)),
                    skew=float(series.skew()),
                    positive_rate=float((series > self.positive_threshold).mean()),
                    negative_rate=float((series < self.negative_threshold).mean()),
                    observation_count=int(series.count()),
                )
                statistics.append(stats_entry)
                column_values.append(values)

            summary = BiasSummary(
                source=source,
                statistics=statistics,
            )

            if statistics:
                stacked = np.concatenate(column_values)
                summary.aggregated_mean = float(np.mean(stacked))
                summary.aggregated_skew = float(pd.Series(stacked).skew())
                if source == "finbert":
                    finbert_values.append(stacked)
                else:
                    vader_values.append(stacked)

            source_payloads[source] = summary

        comparisons: List[BiasComparison] = []
        finbert_array = (
            np.concatenate(finbert_values)
            if finbert_values
            else np.array([], dtype=float)
        )
        vader_array = (
            np.concatenate(vader_values) if vader_values else np.array([], dtype=float)
        )

        if finbert_array.size and vader_array.size:
            mean_diff = float(finbert_array.mean() - vader_array.mean())
            ks_stat: Optional[float] = None
            p_value: Optional[float] = None
            significant: Optional[bool] = None
            if stats is not None:
                ks_statistic, p_value = stats.ks_2samp(finbert_array, vader_array)
                ks_stat = float(ks_statistic)
                significant = bool(p_value < 0.05)
            comparisons.append(
                BiasComparison(
                    reference_source="finbert",
                    comparison_source="vader",
                    mean_difference=mean_diff,
                    ks_statistic=ks_stat,
                    p_value=float(p_value) if p_value is not None else None,
                    significant=significant,
                )
            )

        group_summaries: List[Dict[str, Any]] = []
        if (
            group_col
            and group_col in df.columns
            and finbert_array.size
            and vader_array.size
        ):
            for group_value, group_df in df.groupby(group_col):
                fin_values = self._collect_values(group_df, self.finbert_columns)
                vad_values = self._collect_values(group_df, self.vader_columns)
                if fin_values.size == 0 or vad_values.size == 0:
                    continue
                record = {
                    "group": group_value,
                    "finbert_mean": float(fin_values.mean()),
                    "vader_mean": float(vad_values.mean()),
                    "mean_difference": float(fin_values.mean() - vad_values.mean()),
                }
                if stats is not None:
                    ks_statistic, p_value = stats.ks_2samp(fin_values, vad_values)
                    record["ks_statistic"] = float(ks_statistic)
                    record["p_value"] = float(p_value)
                    record["significant"] = bool(p_value < 0.05)
                group_summaries.append(record)

        return {
            "sources": source_payloads,
            "comparisons": comparisons,
            "group_summaries": group_summaries,
        }

    def _collect_values(self, df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
        values: List[np.ndarray] = []
        for column in columns:
            if column in df.columns:
                series = (
                    df[column].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
                )
                if not series.empty:
                    values.append(series.to_numpy(dtype=float))
        if not values:
            return np.array([], dtype=float)
        return np.concatenate(values)


__all__ = ["SentimentBiasAuditor", "BiasSummary", "BiasComparison", "SourceStatistics"]
