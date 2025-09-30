"""Research-grade evaluation utilities for MEWS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

try:
    from arch import arch_model
except ImportError:  # pragma: no cover - optional dependency
    arch_model = None  # type: ignore

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    import torch as _torch

try:
    from sklearn.metrics import (
        brier_score_loss,
        precision_score,
        roc_auc_score,
    )
except ImportError:  # pragma: no cover - dependency expected
    brier_score_loss = precision_score = roc_auc_score = None  # type: ignore

try:
    from sklearn.calibration import calibration_curve
except ImportError:  # pragma: no cover
    calibration_curve = None  # type: ignore

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None  # type: ignore


@dataclass
class CrisisPeriod:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    @classmethod
    def from_strings(cls, name: str, start: str, end: str) -> "CrisisPeriod":
        return cls(name=name, start=pd.Timestamp(start), end=pd.Timestamp(end))


DEFAULT_CRISES: List[CrisisPeriod] = [
    CrisisPeriod.from_strings("Global Financial Crisis", "2007-07-01", "2009-06-30"),
    CrisisPeriod.from_strings("COVID-19 Shock", "2020-02-01", "2020-12-31"),
    CrisisPeriod.from_strings("Fed Tightening Cycle", "2022-01-01", "2023-12-31"),
]


def precision_at_k(probabilities: np.ndarray, labels: np.ndarray, k: int = 50) -> float:
    if len(probabilities) == 0:
        return float("nan")
    order = np.argsort(probabilities)[::-1]
    top_k = order[: min(k, len(order))]
    return float(np.mean(labels[top_k]))


def baseline_garch_var(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    alpha: float = 0.95,
) -> pd.DataFrame:
    if arch_model is None or stats is None:
        raise ImportError("arch and scipy are required for GARCH baselines")

    results = []
    for symbol, symbol_returns in returns.groupby(level=0):
        symbol_returns = symbol_returns.dropna()
        if len(symbol_returns) < 200:
            continue
        model = arch_model(symbol_returns * 100, vol="GARCH", p=p, q=q)
        fitted = model.fit(disp="off")
        forecasts = fitted.forecast(horizon=1)
        variance = forecasts.variance.iloc[-1, 0] / (100**2)
        mean = forecasts.mean.iloc[-1, 0] / 100
        var_threshold = mean - stats.norm.ppf(alpha) * np.sqrt(variance)
        results.append(
            {
                "Symbol": symbol,
                "VaR": var_threshold,
                "Date": symbol_returns.index[-1][1] if isinstance(symbol_returns.index, pd.MultiIndex) else symbol_returns.index[-1],
            }
        )

    return pd.DataFrame(results)


def baseline_lstm(
    sequences: np.ndarray,
    labels: np.ndarray,
    hidden_dim: int = 64,
    num_layers: int = 1,
    epochs: int = 10,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for LSTM baseline")

    class LSTMRisk(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, layers: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

    def forward(self, x: "_torch.Tensor") -> "_torch.Tensor":
            _, (hn, _) = self.lstm(x)
            return self.head(hn[-1])

    used_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRisk(input_dim=sequences.shape[-1], hidden_dim=hidden_dim, layers=num_layers).to(used_device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(sequences, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(used_device)
            batch_y = batch_y.to(used_device).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(sequences, dtype=torch.float32).to(used_device))
    probabilities = outputs.cpu().numpy().ravel()
    predictions = (probabilities >= 0.5).astype(int)
    return predictions, probabilities


class BenchmarkSuite:
    def __init__(self, crisis_periods: Optional[List[CrisisPeriod]] = None) -> None:
        self.crisis_periods = crisis_periods or DEFAULT_CRISES

    def evaluate_metrics(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        k: int = 50,
    ) -> Dict[str, float]:
        if roc_auc_score is None or precision_score is None or brier_score_loss is None:
            raise ImportError("scikit-learn is required for evaluation metrics")

        if predictions is None:
            predictions = (probabilities >= 0.5).astype(int)

        metrics = {
            "auc": float(roc_auc_score(y_true, probabilities)),
            "brier": float(brier_score_loss(y_true, probabilities)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "precision_at_k": precision_at_k(probabilities, y_true, k=k),
        }
        return metrics

    def evaluate_calibration(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
        bins: int = 10,
    ) -> Dict[str, List[float]]:
        if calibration_curve is None:
            raise ImportError("scikit-learn is required for calibration curves")
        prob_true, prob_pred = calibration_curve(y_true, probabilities, n_bins=bins)
        return {
            "calibration_true": prob_true.tolist(),
            "calibration_pred": prob_pred.tolist(),
        }

    def evaluate_crisis_windows(
        self,
        df: pd.DataFrame,
        date_col: str = "Date",
        label_col: str = "Risk_Label",
        probability_col: str = "Risk_Probability",
    ) -> Dict[str, Dict[str, float]]:
        date_series = pd.to_datetime(df[date_col])
        outcomes = {}
        for crisis in self.crisis_periods:
            mask = (date_series >= crisis.start) & (date_series <= crisis.end)
            if mask.sum() == 0:
                continue
            y_subset = df.loc[mask, label_col].to_numpy()
            p_subset = df.loc[mask, probability_col].to_numpy()
            outcomes[crisis.name] = self.evaluate_metrics(y_subset, p_subset)
        return outcomes

    def compare_models(
        self,
        model_results: Dict[str, Dict[str, float]],
        reference_model: str,
    ) -> Dict[str, Dict[str, float]]:
        if stats is None:
            raise ImportError("scipy is required for statistical tests")

        comparisons: Dict[str, Dict[str, float]] = {}
        ref_metrics = model_results[reference_model]
        for model_name, metrics in model_results.items():
            if model_name == reference_model:
                continue
            auc_diff = metrics["auc"] - ref_metrics["auc"]
            comparisons[model_name] = {
                "auc_difference": auc_diff,
                "significance_pvalue": float(stats.ttest_rel(
                    [metrics["auc"]], [ref_metrics["auc"]]
                ).pvalue),
            }
        return comparisons


class ResearchEvaluator:
    def __init__(
        self,
        crisis_periods: Optional[List[CrisisPeriod]] = None,
        k_precision: int = 50,
    ) -> None:
        self.benchmarks = BenchmarkSuite(crisis_periods=crisis_periods)
        self.k_precision = k_precision

    def evaluate_predictions(
        self,
        df: pd.DataFrame,
        label_col: str = "Risk_Label",
        probability_col: str = "Risk_Probability",
    ) -> Dict[str, Dict[str, float]]:
        y_true = df[label_col].to_numpy()
        probabilities = df[probability_col].to_numpy()
        preds = (probabilities >= 0.5).astype(int)
        metrics = self.benchmarks.evaluate_metrics(y_true, probabilities, preds, k=self.k_precision)
        calibration = self.benchmarks.evaluate_calibration(y_true, probabilities)
        crises = self.benchmarks.evaluate_crisis_windows(df)
        return {
            "overall": metrics,
            "calibration": calibration,
            "crisis_windows": crises,
        }

    def evaluate_against_baselines(
        self,
        mews_results: Dict[str, Dict[str, float]],
        baseline_results: Dict[str, Dict[str, float]],
        reference_model: str = "MEWS Ensemble",
    ) -> Dict[str, Dict[str, float]]:
        combined = {**baseline_results, reference_model: mews_results}
        return self.benchmarks.compare_models(combined, reference_model)


__all__ = [
    "CrisisPeriod",
    "DEFAULT_CRISES",
    "precision_at_k",
    "baseline_garch_var",
    "baseline_lstm",
    "BenchmarkSuite",
    "ResearchEvaluator",
]
