"""Value-at-Risk baselines providing historical and parametric estimates."""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import norm

from .base import BaseBaseline, BaselineResult


class ValueAtRiskBaseline(BaseBaseline):
    """Compute rolling Value-at-Risk thresholds via historical and parametric methods."""

    name = "value_at_risk"

    def __init__(
        self,
        confidence: float = 0.95,
        window: int = 126,
        returns_col: str = "Returns",
    ) -> None:
        super().__init__()
        self.confidence = confidence
        self.window = window
        self.returns_col = returns_col

    def run(
        self,
        df: pd.DataFrame,
        symbol_col: str = "Symbol",
        date_col: str = "Date",
        **_: Any,
    ) -> BaselineResult:
        self._require_columns(df, [symbol_col, date_col, self.returns_col])
        working = self._prepare_dataframe(df, date_col=date_col, symbol_col=symbol_col)

        alpha = 1.0 - self.confidence
        outputs: List[pd.DataFrame] = []
        metadata: Dict[str, Dict[str, Any]] = {}

        for symbol, group in working.groupby(symbol_col):
            series = (
                group.set_index(date_col)[self.returns_col]
                .astype(float)
                .dropna()
            )
            if len(series) < self.window:
                continue

            hist_quantile = series.rolling(self.window).quantile(alpha)
            hist_threshold = (-hist_quantile).clip(lower=0.0)

            rolling_mean = series.rolling(self.window).mean()
            rolling_std = series.rolling(self.window).std(ddof=0)
            param_var = rolling_mean + norm.ppf(alpha) * rolling_std
            param_threshold = (-param_var).clip(lower=0.0)

            def _exceedance_ratio(values: np.ndarray) -> float:
                valid = values[~np.isnan(values)]
                if len(valid) == 0:
                    return math.nan
                threshold = np.quantile(valid, alpha)
                if math.isnan(threshold):
                    return math.nan
                return float(np.mean(valid <= threshold))

            exceedance = series.rolling(self.window).apply(_exceedance_ratio, raw=True)

            combined = pd.DataFrame(
                {
                    symbol_col: symbol,
                    date_col: hist_threshold.index,
                    "var_hist_threshold": hist_threshold.values,
                    "var_param_threshold": param_threshold.values,
                    "var_exceedance_rate": exceedance.values,
                    "var_tail_probability": np.full(len(hist_threshold), alpha),
                }
            )
            outputs.append(combined)

            metadata[symbol] = {
                "observations": int(len(series)),
                "window": self.window,
            }

        if not outputs:
            raise ValueError("Value-at-Risk baseline requires sufficient historical data")

        predictions = pd.concat(outputs, ignore_index=True)
        metadata["confidence_level"] = self.confidence

        return BaselineResult(name=self.name, predictions=predictions, metadata=metadata)
