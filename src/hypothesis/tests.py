"""Statistical hypothesis tests for MEWS research workflows."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .results import (
    GrangerCausalityResult,
    GrangerLagResult,
    LikelihoodRatioResult,
    PairedTestResult,
)

try:  # Optional scientific stack dependencies
    from scipy import stats
except ImportError:  # pragma: no cover - SciPy is an optional dependency
    stats = None  # type: ignore

try:  # pragma: no cover - statsmodels is optional at runtime
    import statsmodels.api as sm  # type: ignore
    from statsmodels.tsa.stattools import grangercausalitytests  # type: ignore
except ImportError:  # pragma: no cover
    sm = None  # type: ignore
    grangercausalitytests = None  # type: ignore


def _ensure_numpy(array: Iterable[float], name: str) -> np.ndarray:
    values = np.asarray(list(array), dtype=float)
    if values.size == 0:
        raise ValueError(f"{name} must contain at least one observation")
    return values


def paired_t_test(
    with_sentiment: Iterable[float],
    without_sentiment: Iterable[float],
    metric_name: str,
    *,
    alpha: float = 0.05,
) -> PairedTestResult:
    """Perform a paired t-test comparing sentiment-enabled and baseline models."""

    if stats is None:
        raise ImportError("scipy is required for the paired t-test")

    sent = _ensure_numpy(with_sentiment, "with_sentiment")
    base = _ensure_numpy(without_sentiment, "without_sentiment")
    if sent.shape != base.shape:
        raise ValueError("with_sentiment and without_sentiment must have matching shapes")

    statistic, p_value = stats.ttest_rel(sent, base, nan_policy="omit")
    diffs = sent - base
    diff_std = float(diffs.std(ddof=1))
    diff_mean = float(diffs.mean())
    effect_size = diff_mean / diff_std if diff_std > 0 else np.nan
    reject = bool(p_value < alpha)

    return PairedTestResult(
        test_name="Paired t-test",
        metric_name=metric_name,
        statistic=float(statistic),
        p_value=float(p_value),
        reject_null=reject,
        alpha=alpha,
        effect_size=effect_size,
        details={
            "mean_with_sentiment": float(np.nanmean(sent)),
            "mean_without_sentiment": float(np.nanmean(base)),
            "mean_difference": diff_mean,
        },
    )


def permutation_test(
    with_sentiment: Iterable[float],
    without_sentiment: Iterable[float],
    metric_name: str,
    *,
    alpha: float = 0.05,
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
) -> PairedTestResult:
    """Run a paired permutation test using sign flipping of metric differences."""

    sent = _ensure_numpy(with_sentiment, "with_sentiment")
    base = _ensure_numpy(without_sentiment, "without_sentiment")
    if sent.shape != base.shape:
        raise ValueError("with_sentiment and without_sentiment must have matching shapes")

    diffs = sent - base
    observed = float(diffs.mean())
    rng = np.random.default_rng(random_state)

    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")

    perm_stats = np.empty(n_permutations, dtype=float)
    for idx in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=diffs.shape[0])
        perm_stats[idx] = float(np.mean(diffs * signs))

    extreme_count = np.sum(np.abs(perm_stats) >= abs(observed))
    p_value = (extreme_count + 1) / (n_permutations + 1)
    diff_std = float(diffs.std(ddof=1))
    effect_size = observed / diff_std if diff_std > 0 else np.nan
    reject = bool(p_value < alpha)

    return PairedTestResult(
        test_name="Permutation test",
        metric_name=metric_name,
        statistic=observed,
        p_value=float(p_value),
        reject_null=reject,
        alpha=alpha,
        effect_size=effect_size,
        details={
            "n_permutations": n_permutations,
            "mean_with_sentiment": float(np.nanmean(sent)),
            "mean_without_sentiment": float(np.nanmean(base)),
            "mean_difference": observed,
        },
    )


def granger_causality(
    *,
    sentiment_series: Sequence[float] | pd.Series,
    fundamental_series: Sequence[float] | pd.Series,
    max_lag: int = 3,
    direction: str = "sentiment->fundamentals",
) -> GrangerCausalityResult:
    """Evaluate if sentiment time-series Granger-causes fundamentals."""

    if grangercausalitytests is None:
        raise ImportError("statsmodels is required for Granger causality testing")

    sentiment = pd.Series(sentiment_series).astype(float)
    fundamentals = pd.Series(fundamental_series).astype(float)
    if sentiment.isna().any() or fundamentals.isna().any():
        raise ValueError("Time-series for Granger causality must not contain NaNs")
    if len(sentiment) != len(fundamentals):
        raise ValueError("Time-series must have equal length")
    if max_lag <= 0:
        raise ValueError("max_lag must be positive")

    data = pd.DataFrame({
        "fundamentals": fundamentals.values,
        "sentiment": sentiment.values,
    })
    test_output = grangercausalitytests(data[["fundamentals", "sentiment"]], maxlag=max_lag, verbose=False)

    lag_results: list[GrangerLagResult] = []
    for lag, results in test_output.items():
        f_test = results[0].get("ssr_ftest") if isinstance(results, tuple) else None
        chi_test = results[0].get("ssr_chi2test") if isinstance(results, tuple) else None
        f_stat, f_pvalue = (float(f_test[0]), float(f_test[1])) if f_test else (np.nan, np.nan)
        chi_stat, chi_pvalue = (float(chi_test[0]), float(chi_test[1])) if chi_test else (None, None)
        lag_results.append(
            GrangerLagResult(
                lag=int(lag),
                f_statistic=f_stat,
                f_pvalue=f_pvalue,
                chi2_statistic=chi_stat,
                chi2_pvalue=chi_pvalue,
            )
        )

    return GrangerCausalityResult(direction=direction, max_lag=max_lag, results=lag_results)


def likelihood_ratio_test(
    *,
    null_model,
    alt_model,
    model_name: str,
    alpha: float = 0.05,
    degrees_freedom: Optional[int] = None,
) -> LikelihoodRatioResult:
    """Compute a likelihood ratio test between two nested models."""

    if stats is None:
        raise ImportError("scipy is required for likelihood ratio testing")

    null_ll, null_df = _extract_model_attributes(null_model)
    alt_ll, alt_df = _extract_model_attributes(alt_model)

    if null_ll is None or alt_ll is None:
        raise ValueError("Models must provide log-likelihood values")

    if degrees_freedom is None:
        if alt_df is None or null_df is None:
            raise ValueError("degrees_freedom must be provided when model degrees are unknown")
        degrees_freedom = int(alt_df - null_df)
    if degrees_freedom <= 0:
        raise ValueError("degrees_freedom must be positive for the likelihood ratio test")

    lr_stat = 2.0 * (alt_ll - null_ll)
    p_value = float(stats.chi2.sf(lr_stat, degrees_freedom))
    reject = bool(p_value < alpha)

    return LikelihoodRatioResult(
        model_name=model_name,
        null_loglike=float(null_ll),
        alt_loglike=float(alt_ll),
        lr_statistic=float(lr_stat),
        degrees_freedom=int(degrees_freedom),
        p_value=p_value,
        reject_null=reject,
        alpha=alpha,
        details={
            "null_params": null_df,
            "alt_params": alt_df,
        },
    )


def _extract_model_attributes(model) -> Tuple[Optional[float], Optional[int]]:
    """Retrieve log-likelihood and degrees of freedom from common model objects."""

    loglike = None
    df_model = None

    if hasattr(model, "llf"):
        loglike = float(getattr(model, "llf"))
    elif isinstance(model, (tuple, list)) and len(model) >= 1:
        loglike = float(model[0])

    if model is not None:
        if hasattr(model, "df_model"):
            df_model = int(getattr(model, "df_model"))
        elif hasattr(model, "df_resid") and hasattr(model, "nobs"):
            df_model = int(getattr(model, "nobs")) - int(getattr(model, "df_resid"))
        elif isinstance(model, (tuple, list)) and len(model) >= 2:
            df_model = int(model[1])

    return loglike, df_model


__all__ = [
    "paired_t_test",
    "permutation_test",
    "granger_causality",
    "likelihood_ratio_test",
]
