"""Adaptive ensemble methods for market regime-aware risk prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover - SKLearn required for evaluation
    roc_auc_score = None  # type: ignore


@dataclass
class RegimeWeights:
    """Container for storing per-regime ensemble weights."""

    regime: str
    weights: Dict[str, float]


class VolatilityRegimeDetector:
    """Detect market regimes using rolling realised volatility."""

    def __init__(
        self,
        lookback: int = 30,
        low_vol_quantile: float = 0.33,
        high_vol_quantile: float = 0.66,
        returns_col: str = "Returns",
        date_col: str = "Date",
        symbol_col: str = "Symbol",
    ) -> None:
        self.lookback = lookback
        self.low_vol_quantile = low_vol_quantile
        self.high_vol_quantile = high_vol_quantile
        self.returns_col = returns_col
        self.date_col = date_col
        self.symbol_col = symbol_col
        self._volatility_series: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame) -> "VolatilityRegimeDetector":
        if self.returns_col not in df.columns:
            raise ValueError(
                f"Column '{self.returns_col}' is required for volatility regime detection"
            )

        working = df.copy()
        working[self.date_col] = pd.to_datetime(working[self.date_col])
        working.sort_values([self.symbol_col, self.date_col], inplace=True)

        rolling_vol = (
            working.groupby(self.symbol_col)[self.returns_col]
            .rolling(window=self.lookback, min_periods=max(5, self.lookback // 3))
            .std()
            .reset_index(level=0, drop=True)
        )
        self._volatility_series = rolling_vol
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        if self._volatility_series is None:
            self.fit(df)

        vol = self._volatility_series.loc[df.index]
        low_threshold = vol.quantile(self.low_vol_quantile)
        high_threshold = vol.quantile(self.high_vol_quantile)

        regimes = pd.Series(index=df.index, dtype="object")
        regimes.loc[vol <= low_threshold] = "low_volatility"
        regimes.loc[(vol > low_threshold) & (vol < high_threshold)] = "moderate_volatility"
        regimes.loc[vol >= high_threshold] = "high_volatility"
        regimes.fillna("moderate_volatility", inplace=True)
        return regimes

    def fit_transform(self, df: pd.DataFrame) -> pd.Series:
        self.fit(df)
        return self.transform(df)


class RegimeAdaptiveEnsemble:
    """Combines model probabilities with regime-aware weights."""

    def __init__(self, floor_weight: float = 1e-3) -> None:
        self.floor_weight = floor_weight
        self.regime_weights: Dict[str, Dict[str, float]] = {}
        self.models: List[str] = []
        self.default_weights: Dict[str, float] = {}

    def fit(
        self,
        model_probabilities: Mapping[str, Iterable[float]],
        y_true: Iterable[int],
        regimes: Iterable[str],
    ) -> "RegimeAdaptiveEnsemble":
        if roc_auc_score is None:
            raise ImportError("scikit-learn is required to compute regime weights")

        probabilities_df = pd.DataFrame(model_probabilities)
        probabilities_df.index = pd.Index(range(len(probabilities_df)))
        y_array = np.asarray(list(y_true))
        regime_series = pd.Series(list(regimes), index=probabilities_df.index)

        self.models = list(probabilities_df.columns)

        # baseline weights using full-sample performance
        baseline_scores = {}
        for model_name in self.models:
            score = roc_auc_score(y_array, probabilities_df[model_name])
            baseline_scores[model_name] = max(score, self.floor_weight)
        baseline_total = sum(baseline_scores.values())
        self.default_weights = {
            model: baseline_scores[model] / baseline_total for model in self.models
        }

        for regime, regime_idx in regime_series.groupby(regime_series).groups.items():
            regime_mask = regime_series.index.isin(regime_idx)
            if regime_mask.sum() < max(30, int(0.05 * len(regime_series))):
                self.regime_weights[regime] = self.default_weights.copy()
                continue

            regime_scores = {}
            for model_name in self.models:
                try:
                    score = roc_auc_score(
                        y_array[regime_mask], probabilities_df.loc[regime_mask, model_name]
                    )
                except ValueError:
                    score = self.floor_weight
                regime_scores[model_name] = max(score, self.floor_weight)

            score_sum = sum(regime_scores.values())
            self.regime_weights[regime] = {
                model: regime_scores[model] / score_sum for model in self.models
            }

        return self

    def predict(
        self,
        model_probabilities: Mapping[str, Iterable[float]],
        regimes: Iterable[str],
    ) -> np.ndarray:
        probabilities_df = pd.DataFrame(model_probabilities)
        probabilities_df.index = pd.Index(range(len(probabilities_df)))
        regime_series = pd.Series(list(regimes), index=probabilities_df.index)

        combined = np.zeros(len(probabilities_df))
        for idx, row in probabilities_df.iterrows():
            regime = regime_series.at[idx]
            weights = self.regime_weights.get(regime, self.default_weights)
            total = sum(weights.get(model, 0.0) * row[model] for model in self.models)
            combined[idx] = total
        return combined

    def to_json(self) -> Dict[str, Dict[str, float]]:
        payload: Dict[str, Dict[str, float]] = {
            "default": self.default_weights,
            **self.regime_weights,
        }
        return payload


__all__ = [
    "RegimeAdaptiveEnsemble",
    "VolatilityRegimeDetector",
    "RegimeWeights",
]
