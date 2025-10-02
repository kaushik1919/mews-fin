"""Hypothesis testing and ablation utilities for MEWS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
except ImportError:  # pragma: no cover - optional dependency
    sm = None  # type: ignore

try:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover
    roc_auc_score = None  # type: ignore
    train_test_split = None  # type: ignore

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None  # type: ignore


@dataclass
class HypothesisResult:
    null_model_ll: float
    alt_model_ll: float
    lr_statistic: float
    p_value: float
    reject_null: bool


class SentimentImpactTester:
    def __init__(self, target_col: str = "Risk_Label") -> None:
        self.target_col = target_col

    def run_test(
        self,
        df: pd.DataFrame,
        fundamental_features: Iterable[str],
        sentiment_features: Iterable[str],
    ) -> HypothesisResult:
        if sm is None or stats is None:
            raise ImportError(
                "statsmodels and scipy are required for hypothesis testing"
            )

        y = df[self.target_col].astype(int)
        X_base = sm.add_constant(df[list(fundamental_features)], has_constant="add")
        X_full = sm.add_constant(
            df[list(fundamental_features) + list(sentiment_features)],
            has_constant="add",
        )

        null_model = sm.Logit(y, X_base).fit(disp=False)
        alt_model = sm.Logit(y, X_full).fit(disp=False)

        lr_stat = 2 * (alt_model.llf - null_model.llf)
        df_diff = X_full.shape[1] - X_base.shape[1]
        p_value = float(stats.chi2.sf(lr_stat, df_diff))
        reject = p_value < 0.05

        return HypothesisResult(
            null_model_ll=float(null_model.llf),
            alt_model_ll=float(alt_model.llf),
            lr_statistic=float(lr_stat),
            p_value=p_value,
            reject_null=reject,
        )


class GraphFeatureAblation:
    def __init__(self, target_col: str = "Risk_Label") -> None:
        self.target_col = target_col

    def evaluate(
        self,
        df: pd.DataFrame,
        graph_feature_prefix: str = "graph_",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, float]:
        if roc_auc_score is None or train_test_split is None:
            raise ImportError("scikit-learn is required for ablation study")

        feature_cols = [
            col for col in df.columns if col not in {self.target_col, "Date", "Symbol"}
        ]
        graph_features = [
            col for col in feature_cols if col.startswith(graph_feature_prefix)
        ]
        non_graph_features = [col for col in feature_cols if col not in graph_features]

        X_graph = df[non_graph_features + graph_features].fillna(0)
        X_no_graph = df[non_graph_features].fillna(0)
        y = df[self.target_col].astype(int)

        Xg_train, Xg_test, yg_train, yg_test = train_test_split(
            X_graph, y, test_size=test_size, random_state=random_state, stratify=y
        )
        Xng_train, Xng_test, yng_train, yng_test = train_test_split(
            X_no_graph, y, test_size=test_size, random_state=random_state, stratify=y
        )

        from sklearn.linear_model import LogisticRegression

        model_graph = LogisticRegression(max_iter=300, class_weight="balanced")
        model_graph.fit(Xg_train, yg_train)
        model_no_graph = LogisticRegression(max_iter=300, class_weight="balanced")
        model_no_graph.fit(Xng_train, yng_train)

        auc_with_graph = roc_auc_score(
            yg_test, model_graph.predict_proba(Xg_test)[:, 1]
        )
        auc_without_graph = roc_auc_score(
            yng_test, model_no_graph.predict_proba(Xng_test)[:, 1]
        )

        return {
            "auc_with_graph": float(auc_with_graph),
            "auc_without_graph": float(auc_without_graph),
            "auc_difference": float(auc_with_graph - auc_without_graph),
        }


__all__ = [
    "HypothesisResult",
    "SentimentImpactTester",
    "GraphFeatureAblation",
]
