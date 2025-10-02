"""Tests for the hypothesis testing suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.hypothesis import (
    HypothesisReportBuilder,
    granger_causality,
    likelihood_ratio_test,
    paired_t_test,
    permutation_test,
)

try:
    import statsmodels.api as sm  # type: ignore
except ImportError:  # pragma: no cover
    sm = None  # type: ignore


@pytest.mark.skipif(sm is None, reason="statsmodels is required for these tests")
def test_paired_tests_detect_difference():
    rng = np.random.default_rng(123)
    without_sentiment = rng.normal(loc=0.70, scale=0.02, size=50)
    with_sentiment = without_sentiment + 0.05

    t_result = paired_t_test(with_sentiment, without_sentiment, metric_name="AUC")
    perm_result = permutation_test(
        with_sentiment,
        without_sentiment,
        metric_name="AUC",
        n_permutations=500,
        random_state=7,
    )

    assert t_result.reject_null and t_result.p_value < 0.001
    assert perm_result.reject_null and perm_result.p_value < 0.01
    assert pytest.approx(t_result.details["mean_difference"], rel=1e-5) == 0.05


@pytest.mark.skipif(sm is None, reason="statsmodels is required for these tests")
def test_granger_causality_identifies_signal():
    rng = np.random.default_rng(42)
    n = 200
    sentiment = rng.normal(size=n)
    fundamentals = np.zeros(n)
    for idx in range(1, n):
        fundamentals[idx] = (
            0.6 * fundamentals[idx - 1]
            + 0.4 * sentiment[idx - 1]
            + rng.normal(scale=0.5)
        )

    result = granger_causality(
        sentiment_series=sentiment,
        fundamental_series=fundamentals,
        max_lag=2,
    )

    assert result.results
    assert any(lag_result.f_pvalue < 0.05 for lag_result in result.results)


@pytest.mark.skipif(sm is None, reason="statsmodels is required for these tests")
def test_likelihood_ratio_uses_statsmodels_results():
    rng = np.random.default_rng(55)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logits = -0.3 + 1.2 * x1 + 0.9 * x2
    probs = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probs)

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    null_model = sm.Logit(df["y"], sm.add_constant(df[["x1"]])).fit(disp=False)
    alt_model = sm.Logit(df["y"], sm.add_constant(df[["x1", "x2"]])).fit(disp=False)

    result = likelihood_ratio_test(
        null_model=null_model,
        alt_model=alt_model,
        model_name="Sentiment feature",
    )

    assert result.reject_null
    assert result.p_value < 0.05
    assert result.degrees_freedom == 1


@pytest.mark.skipif(sm is None, reason="statsmodels is required for these tests")
def test_report_builder_creates_outputs(tmp_path: Path):
    rng = np.random.default_rng(100)
    base = rng.normal(loc=0.75, scale=0.01, size=30)
    enhanced = base + 0.02
    paired_result = paired_t_test(enhanced, base, metric_name="AUC")

    granger_result = granger_causality(
        sentiment_series=np.arange(50, dtype=float),
        fundamental_series=np.arange(50, dtype=float) + rng.normal(scale=0.1, size=50),
        max_lag=1,
    )

    rng = np.random.default_rng(21)
    df = pd.DataFrame(
        {
            "y": rng.binomial(1, 0.5, size=120),
            "x1": rng.normal(size=120),
            "x2": rng.normal(size=120),
        }
    )
    null_model = sm.Logit(df["y"], sm.add_constant(df[["x1"]])).fit(disp=False)
    alt_model = sm.Logit(df["y"], sm.add_constant(df[["x1", "x2"]])).fit(disp=False)
    lr_result = likelihood_ratio_test(
        null_model=null_model,
        alt_model=alt_model,
        model_name="Likelihood",
    )

    builder = HypothesisReportBuilder(output_dir=str(tmp_path))
    outputs = builder.build_reports(
        paired_results=[paired_result],
        granger_results=[granger_result],
        lr_results=[lr_result],
        metadata={"Dataset": "Unit Test"},
        base_filename="unit_test_report",
    )

    markdown_path = outputs["markdown"]
    html_path = outputs["html"]
    assert markdown_path.exists()
    assert html_path.exists()
    assert "Paired Model Comparisons" in markdown_path.read_text(encoding="utf-8")
    assert "Hypothesis Testing Summary" in html_path.read_text(encoding="utf-8")
