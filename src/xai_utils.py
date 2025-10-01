"""Explainable AI utilities for MEWS ensemble models."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

try:  # Optional dependency
    import shap  # type: ignore[import]
except ImportError:  # pragma: no cover - shap is optional
    shap = None  # type: ignore

SHAP_AVAILABLE = shap is not None

try:  # Optional dependency
    from lime.lime_tabular import LimeTabularExplainer  # type: ignore[import]
except ImportError:  # pragma: no cover - lime is optional
    LimeTabularExplainer = None  # type: ignore

LIME_AVAILABLE = LimeTabularExplainer is not None


def compute_global_shap_importance(
    model: Any, X: pd.DataFrame, max_features: int = 25
) -> Optional[pd.DataFrame]:
    """Compute global SHAP feature importance.

    Args:
        model: Fitted estimator supporting ``predict_proba`` or ``predict``.
        X: Feature matrix used for explanations.
        max_features: Maximum number of features to keep for visualization.

    Returns:
        DataFrame with columns ``feature`` and ``importance`` sorted descending.
    """

    if shap is None:
        LOGGER.warning("SHAP library not available; skipping global importance")
        return None

    if X.empty:
        LOGGER.warning("Empty feature matrix passed to SHAP computation")
        return None

    try:
        explainer = shap.Explainer(model.predict_proba, X, feature_names=X.columns)
        shap_values = explainer(X)
    except Exception as exc:  # pragma: no cover - defensive against SHAP failures
        LOGGER.exception("Failed to compute global SHAP values: %s", exc)
        return None
    values = shap_values.values

    feature_count = X.shape[1]

    if values.ndim == 3:
        if values.shape[1] == feature_count:  # (samples, features, outputs)
            importance = np.abs(values).mean(axis=(0, 2))
        elif values.shape[2] == feature_count:  # (samples, outputs, features)
            importance = np.abs(values).mean(axis=(0, 1))
        else:
            reshaped = values.reshape(values.shape[0], -1, feature_count)
            importance = np.abs(reshaped).mean(axis=(0, 1))
    else:
        importance = np.abs(values).mean(axis=0)

    if importance.ndim > 1:
        importance = importance.reshape(-1)
    summary = (
        pd.DataFrame({"feature": X.columns, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(max_features)
        .reset_index(drop=True)
    )
    return summary


def compute_local_shap_explanation(
    model: Any, X: pd.DataFrame, sample_index: int
) -> Optional[pd.DataFrame]:
    """Compute local SHAP values for a single observation."""

    if shap is None:
        LOGGER.warning("SHAP library not available; skipping local explanation")
        return None

    if X.empty:
        LOGGER.warning("Empty feature matrix passed to SHAP computation")
        return None

    index = np.clip(sample_index, 0, len(X) - 1)
    try:
        explainer = shap.Explainer(model.predict_proba, X, feature_names=X.columns)
        shap_values = explainer(X.iloc[[index]])
    except Exception as exc:  # pragma: no cover - defensive against SHAP failures
        LOGGER.exception("Failed to compute local SHAP values: %s", exc)
        return None
    values = shap_values.values[0]

    feature_count = X.shape[1]

    if values.ndim == 2:
        if values.shape[0] == feature_count:
            values = values.mean(axis=1)
        elif values.shape[1] == feature_count:
            values = values.mean(axis=0)
        else:
            values = values.reshape(-1, feature_count).mean(axis=0)

    if values.ndim > 1:
        values = values.reshape(-1)

    base = shap_values.base_values[0]
    if np.ndim(base) > 0:
        base = float(np.array(base).mean())

    contribution = pd.DataFrame(
        {
            "feature": X.columns,
            "shap_value": values,
            "abs_shap": np.abs(values),
            "base_value": base,
        }
    ).sort_values("abs_shap", ascending=False)
    return contribution


def compute_lime_explanation(
    model: Any,
    X: pd.DataFrame,
    sample_index: int,
    class_names: Optional[np.ndarray] = None,
    num_features: int = 10,
) -> Optional[pd.DataFrame]:
    """Generate local explanations using LIME."""

    if LimeTabularExplainer is None:
        LOGGER.warning("LIME library not available; skipping explanation")
        return None

    if X.empty:
        LOGGER.warning("Empty feature matrix passed to LIME computation")
        return None

    try:
        explainer = LimeTabularExplainer(
            training_data=X.values,
            feature_names=list(X.columns),
            class_names=class_names,
            verbose=False,
            mode="classification",
        )
        index = np.clip(sample_index, 0, len(X) - 1)
        explanation = explainer.explain_instance(
            X.values[index], model.predict_proba, num_features=num_features
        )
    except Exception as exc:  # pragma: no cover - defensive against LIME failures
        LOGGER.exception("Failed to compute LIME explanation: %s", exc)
        return None
    explanation_df = pd.DataFrame(
        explanation.as_list(), columns=["feature", "weight"]
    ).assign(abs_weight=lambda df: df["weight"].abs())
    return explanation_df


__all__ = [
    "SHAP_AVAILABLE",
    "LIME_AVAILABLE",
    "compute_global_shap_importance",
    "compute_local_shap_explanation",
    "compute_lime_explanation",
]
