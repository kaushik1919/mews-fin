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

try:  # Optional dependency
    from lime.lime_tabular import LimeTabularExplainer  # type: ignore[import]
except ImportError:  # pragma: no cover - lime is optional
    LimeTabularExplainer = None  # type: ignore


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

    explainer = shap.Explainer(model.predict_proba, X, feature_names=X.columns)
    shap_values = explainer(X)
    importance = np.abs(shap_values.values).mean(axis=0)
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
    explainer = shap.Explainer(model.predict_proba, X, feature_names=X.columns)
    shap_values = explainer(X.iloc[[index]])
    values = shap_values.values[0]
    contribution = pd.DataFrame(
        {
            "feature": X.columns,
            "shap_value": values,
            "abs_shap": np.abs(values),
            "base_value": shap_values.base_values[0],
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
    explanation_df = pd.DataFrame(
        explanation.as_list(), columns=["feature", "weight"]
    ).assign(abs_weight=lambda df: df["weight"].abs())
    return explanation_df


__all__ = [
    "compute_global_shap_importance",
    "compute_local_shap_explanation",
    "compute_lime_explanation",
]
