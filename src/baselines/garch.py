"""GARCH-based volatility and risk threshold baseline."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import norm

from .base import BaseBaseline, BaselineResult

try:  # pragma: no cover - optional dependency
    from arch import arch_model  # type: ignore
except ImportError:  # pragma: no cover
    arch_model = None  # type: ignore


class GARCHBaseline(BaseBaseline):
    """Estimate risk thresholds using a GARCH(1,1) volatility model."""

    name = "garch_1_1"

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        dist: str = "normal",
        confidence: float = 0.95,
        min_observations: int = 100,
        returns_col: str = "Returns",
    ) -> None:
        super().__init__()
        self.p = p
        self.q = q
        self.dist = dist
        self.confidence = confidence
        self.min_observations = min_observations
        self.returns_col = returns_col

    def run(
        self,
        df: pd.DataFrame,
        symbol_col: str = "Symbol",
        date_col: str = "Date",
        **_: Any,
    ) -> BaselineResult:
        if arch_model is None:
            raise ImportError("arch package is required for the GARCH baseline")

        self._require_columns(df, [symbol_col, date_col, self.returns_col])
        working = self._prepare_dataframe(df, date_col=date_col, symbol_col=symbol_col)

        outputs: List[pd.DataFrame] = []
        metadata: Dict[str, Dict[str, Any]] = {}
        z_score = float(norm.ppf(self.confidence))
        epsilon = 1e-8

        for symbol, group in working.groupby(symbol_col):
            series = group.set_index(date_col)[self.returns_col].dropna().astype(float)
            if len(series) < self.min_observations or series.var() <= 0:
                self.logger.debug(
                    "Skipping symbol %s due to insufficient data for GARCH baseline",
                    symbol,
                )
                continue

            try:
                scaled = (
                    series * 100.0
                )  # GARCH expects percentage returns for stability
                model = arch_model(
                    scaled,
                    vol="Garch",
                    p=self.p,
                    q=self.q,
                    mean="Zero",
                    dist=self.dist,
                )
                fit_res = model.fit(update_freq=0, disp="off")
            except Exception as exc:  # pragma: no cover - defensive
                logging.getLogger(__name__).warning(
                    "GARCH baseline failed for %s: %s", symbol, exc
                )
                continue

            cond_vol = fit_res.conditional_volatility / 100.0
            cond_vol = cond_vol.reindex(series.index)
            tail_ratio = np.abs(series) / np.maximum(cond_vol, epsilon)
            tail_prob = np.clip(norm.sf(tail_ratio) * 2.0, 0.0, 1.0)
            var_threshold = np.maximum(z_score * cond_vol, 0.0)

            result_df = pd.DataFrame(
                {
                    symbol_col: symbol,
                    date_col: cond_vol.index,
                    "garch_volatility": cond_vol.values,
                    "garch_var_threshold": var_threshold.values,
                    "garch_tail_probability": tail_prob,
                }
            )
            outputs.append(result_df)
            metadata[symbol] = {
                "bic": float(fit_res.bic),
                "aic": float(fit_res.aic),
                "converged": bool(getattr(fit_res, "converged", True)),
                "loglikelihood": float(getattr(fit_res, "loglikelihood", np.nan)),
            }

        if not outputs:
            raise ValueError("GARCH baseline could not be computed for any symbols")

        predictions = pd.concat(outputs, ignore_index=True)
        metadata["confidence_level"] = self.confidence
        metadata["min_observations"] = self.min_observations

        return BaselineResult(
            name=self.name, predictions=predictions, metadata=metadata
        )
