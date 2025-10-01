"""Regime-aware ensemble strategies for MEWS."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, MutableMapping, Optional

import warnings

import numpy as np
import pandas as pd

from .base import BaseEnsemble

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:  # pragma: no cover - optional dependency
    LogisticRegression = None  # type: ignore

try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover - sklearn expected
    roc_auc_score = None  # type: ignore


class VolatilityRegimeDetector:
    """Detect market regimes based on rolling realised volatility."""

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


class RegimeAdaptiveEnsemble(BaseEnsemble):
    """Regime-aware combination of model probabilities with optional meta-modeling."""

    def __init__(
        self,
        floor_weight: float = 1e-3,
        lookback: int = 30,
        low_vol_quantile: float = 0.33,
        high_vol_quantile: float = 0.66,
        returns_col: str = "Returns",
        date_col: str = "Date",
        symbol_col: str = "Symbol",
        min_regime_fraction: float = 0.05,
        min_regime_samples: int = 30,
        use_meta_model: bool = False,
        meta_model_penalty: float = 1.0,
        meta_model_solver: str = "lbfgs",
        meta_model_max_iter: int = 1000,
    ) -> None:
        super().__init__()
        self.floor_weight = floor_weight
        self.min_regime_fraction = float(min_regime_fraction)
        self.min_regime_samples = int(min_regime_samples)
        self.detector = VolatilityRegimeDetector(
            lookback=lookback,
            low_vol_quantile=low_vol_quantile,
            high_vol_quantile=high_vol_quantile,
            returns_col=returns_col,
            date_col=date_col,
            symbol_col=symbol_col,
        )
        self.regime_weights = {}
        self.default_weights = {}
        self.meta_models = {}
        self.meta_model_penalty = meta_model_penalty
        self.meta_model_solver = meta_model_solver
        self.meta_model_max_iter = meta_model_max_iter
        self.use_meta_model = bool(use_meta_model and LogisticRegression is not None)
        if use_meta_model and LogisticRegression is None:  # pragma: no cover - diagnostics only
            warnings.warn(
                "Meta-model requested for RegimeAdaptiveEnsemble but scikit-learn is not available."
                " Falling back to score-based weighting.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._last_regimes = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        model_probabilities: Mapping[str, Iterable[float]],
        y_true: Iterable[int],
        metadata: Optional[pd.DataFrame] = None,
        regimes: Optional[Iterable[str]] = None,
        **_: object,
    ) -> "RegimeAdaptiveEnsemble":
        names, stacked = self._stack_probabilities(model_probabilities)
        probabilities_df = pd.DataFrame(stacked.T, columns=names)
        y_array = np.asarray(list(y_true), dtype=float)
        if len(probabilities_df) != len(y_array):
            raise ValueError("y_true length must match number of probability rows")

        regime_series = self._resolve_regimes(probabilities_df.index, metadata, regimes)
        self._last_regimes = regime_series.copy()

        baseline_scores = self._compute_global_scores(probabilities_df, y_array, names)
        self.default_weights = baseline_scores
        self.weights = dict(self.default_weights)

        self.regime_weights = {}
        self.meta_models = {}

        for regime_value, mask, sufficient in self._iter_regime_masks(regime_series):
            regime_probs = probabilities_df.loc[mask]
            regime_targets = y_array[mask]
            if len(regime_probs) == 0:
                continue

            if not sufficient:
                self.regime_weights[regime_value] = self.default_weights
                continue

            weights = self._compute_regime_weights(names, regime_probs, regime_targets)

            if self.use_meta_model and len(np.unique(regime_targets)) > 1:
                meta_weights = self._train_meta_model(regime_value, names, regime_probs, regime_targets)
                if meta_weights is not None:
                    weights = meta_weights

            self.regime_weights[regime_value] = weights

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        model_probabilities: Mapping[str, Iterable[float]],
        metadata: Optional[pd.DataFrame] = None,
        regimes: Optional[Iterable[str]] = None,
        **_: object,
    ) -> np.ndarray:
        names, stacked = self._stack_probabilities(model_probabilities)
        probabilities_df = pd.DataFrame(stacked.T, columns=names)
        regime_series = self._resolve_regimes(probabilities_df.index, metadata, regimes)

        predictions = np.zeros(len(probabilities_df), dtype=float)

        for regime_value, mask, sufficient in self._iter_regime_masks(regime_series):
            subset = probabilities_df.loc[mask]
            if subset.empty:
                continue

            mask_array = mask.to_numpy()

            if self.use_meta_model and regime_value in self.meta_models:
                model = self.meta_models[regime_value]
                predictions[mask_array] = model.predict_proba(subset.values)[:, 1]
                continue

            weights = self.regime_weights.get(regime_value, self.default_weights)
            weight_vector = self._vector_from_mapping(names, weights)
            weight_vector = self._normalize_weight_vector(weight_vector)
            predictions[mask_array] = np.dot(subset.values, weight_vector)

        return predictions

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def to_json(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "default": dict(self.default_weights),
            "regimes": {regime: dict(weights) for regime, weights in self.regime_weights.items()},
        }
        payload["meta_model"] = {
            "enabled": bool(self.meta_models),
        }
        if self.meta_models:
            payload["meta_model"]["coefficients"] = {
                regime: {
                    name: float(coef)
                    for name, coef in zip(self.model_names, model.coef_[0])
                }
                for regime, model in self.meta_models.items()
            }
        return payload

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _resolve_regimes(
        self,
        index: pd.Index,
        metadata: Optional[pd.DataFrame],
        regimes: Optional[Iterable[str]],
    ) -> pd.Series:
        if metadata is not None:
            if len(metadata) != len(index):
                raise ValueError("Metadata length must match model probabilities when detecting regimes")
            meta = metadata.reset_index(drop=True).copy()
            meta.index = index
            meta = self._ensure_returns_column(meta)
            regime_series = self.detector.fit_transform(meta)
            return regime_series

        if regimes is not None:
            regime_series = pd.Series(list(regimes), index=index)
            return regime_series

        if self._last_regimes is not None:
            return self._last_regimes.reindex(index, fill_value="moderate_volatility")

        raise ValueError("metadata or regimes must be provided to determine volatility regimes")

    def _iter_regime_masks(self, regime_series: pd.Series):
        total = len(regime_series)
        min_samples = max(self.min_regime_samples, int(self.min_regime_fraction * total))
        for regime_value in regime_series.unique():
            mask = regime_series == regime_value
            yield regime_value, mask, mask.sum() >= min_samples

    def _compute_global_scores(
        self,
        probabilities_df: pd.DataFrame,
        y_array: np.ndarray,
        names: Iterable[str],
    ) -> Mapping[str, float]:
        names = tuple(names)
        if roc_auc_score is None:
            weights = np.ones(len(names), dtype=float)
            weights = self._normalize_weight_vector(weights)
            return {name: float(weight) for name, weight in zip(names, weights)}

        baseline_scores = {}
        for model_name in names:
            try:
                score = float(roc_auc_score(y_array, probabilities_df[model_name]))
            except ValueError:
                score = 0.5
            baseline_scores[model_name] = max(score, self.floor_weight)

        total = sum(baseline_scores.values())
        normalized = {name: baseline_scores[name] / total for name in names}
        return normalized

    def _compute_regime_weights(
        self,
        names: Iterable[str],
        regime_probs: pd.DataFrame,
        regime_targets: np.ndarray,
    ) -> Mapping[str, float]:
        names = tuple(names)
        if roc_auc_score is None:
            return self.default_weights

        regime_scores = {}
        for model_name in names:
            try:
                score = float(roc_auc_score(regime_targets, regime_probs[model_name]))
            except ValueError:
                score = self.floor_weight
            regime_scores[model_name] = max(score, self.floor_weight)

        score_sum = sum(regime_scores.values())
        weights = {model: regime_scores[model] / score_sum for model in names}
        return weights

    def _train_meta_model(
        self,
        regime: str,
        names: Iterable[str],
        regime_probs: pd.DataFrame,
        regime_targets: np.ndarray,
    ) -> Optional[Mapping[str, float]]:
        names = tuple(names)
        if LogisticRegression is None:
            return None

        model = LogisticRegression(
            penalty="l2",
            C=float(self.meta_model_penalty),
            solver=self.meta_model_solver,
            max_iter=self.meta_model_max_iter,
        )
        try:
            model.fit(regime_probs.values, regime_targets.astype(int))
        except Exception:  # pragma: no cover - fallback path
            return None

        coefs = np.abs(model.coef_[0])
        if not np.any(coefs):
            return None

        weight_vector = self._normalize_weight_vector(coefs)
        weights = {name: float(weight) for name, weight in zip(names, weight_vector)}
        self.meta_models[regime] = model
        return weights

    def _ensure_returns_column(self, metadata: pd.DataFrame) -> pd.DataFrame:
        returns_col = self.detector.returns_col
        if returns_col in metadata.columns:
            return metadata

        price_candidates = [
            col for col in ["Close", "Adj Close", "close", "adj_close", "price"] if col in metadata.columns
        ]
        if not price_candidates:
            raise ValueError(
                f"Returns column '{returns_col}' not found and unable to infer from price columns"
            )

        price_col = price_candidates[0]
        working = metadata.copy()
        working = working.sort_values([self.detector.symbol_col, self.detector.date_col])
        working[returns_col] = (
            working.groupby(self.detector.symbol_col)[price_col].pct_change().fillna(0.0)
        )
        working = working.sort_index()
        metadata[returns_col] = working.loc[metadata.index, returns_col]
        return metadata
