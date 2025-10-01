"""Probability calibration utilities and reliability diagram helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    from sklearn.calibration import calibration_curve
    from sklearn.linear_model import LogisticRegression
except ImportError:  # pragma: no cover
    calibration_curve = None  # type: ignore
    LogisticRegression = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore


@dataclass
class CalibrationCurve:
    probabilities_true: np.ndarray
    probabilities_pred: np.ndarray

    def as_dict(self) -> Dict[str, Iterable[float]]:
        return {
            "calibration_true": self.probabilities_true.tolist(),
            "calibration_pred": self.probabilities_pred.tolist(),
        }


class ProbabilityCalibrator:
    """Logistic calibration (a.k.a Platt scaling) for probability outputs."""

    def __init__(self) -> None:
        if LogisticRegression is None:  # pragma: no cover - defensive
            raise ImportError("scikit-learn is required for probability calibration")
        self._model = LogisticRegression()

    def fit(self, probabilities: np.ndarray, targets: np.ndarray) -> "ProbabilityCalibrator":
        probs = np.asarray(probabilities).reshape(-1, 1)
        labels = np.asarray(targets).astype(int)
        self._model.fit(probs, labels)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        probs = np.asarray(probabilities).reshape(-1, 1)
        calibrated = self._model.predict_proba(probs)[:, 1]
        return calibrated

    def calibration_curve(self, probabilities: np.ndarray, targets: np.ndarray, bins: int = 10) -> CalibrationCurve:
        if calibration_curve is None:
            raise ImportError("scikit-learn is required for calibration curves")
        prob_true, prob_pred = calibration_curve(targets, probabilities, n_bins=bins)
        return CalibrationCurve(probabilities_true=prob_true, probabilities_pred=prob_pred)


def compute_calibration_curve(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    bins: int = 10,
) -> CalibrationCurve:
    calibrator = ProbabilityCalibrator()
    calibrator.fit(probabilities, targets)
    return calibrator.calibration_curve(probabilities, targets, bins=bins)


def save_reliability_diagram(
    curve: CalibrationCurve,
    *,
    path: str,
    title: str = "Reliability Diagram",
) -> str:
    if plt is None:
        raise ImportError("matplotlib is required to plot reliability diagrams")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax.plot(curve.probabilities_pred, curve.probabilities_true, marker="o", label="Model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
