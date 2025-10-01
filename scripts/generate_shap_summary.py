"""Generate a SHAP summary plot for the latest trained model."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt


DATASET_PATH = Path("data/dataset_with_risk_labels.csv")
MODEL_ROOT = Path("models")
OUTPUT_PATH = Path("outputs/shap_summary.png")
TARGET_COLUMN = "Risk_Label"


def _latest_model_dir() -> Path:
    candidates = [path for path in MODEL_ROOT.glob("models_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError("No saved model directories found under 'models/'.")
    return max(candidates, key=lambda path: path.name)


def _load_feature_matrix(df: pd.DataFrame, feature_names: np.ndarray | None) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if TARGET_COLUMN in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[TARGET_COLUMN])
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if feature_names is None:
        return numeric_df

    missing = [name for name in feature_names if name not in numeric_df.columns]
    if missing:
        raise KeyError(
            "Missing expected features in dataset: " + ", ".join(sorted(missing))
        )

    return numeric_df.loc[:, feature_names]


def main() -> int:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset '{DATASET_PATH}' is required to generate the SHAP summary."
        )

    df = pd.read_csv(DATASET_PATH)
    if df.empty:
        raise ValueError("The dataset_with_risk_labels.csv file is empty.")

    model_dir = _latest_model_dir()
    model_path = model_dir / "random_forest_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Random forest model not found at '{model_path}'. Ensure the training pipeline has run."
        )

    model = joblib.load(model_path)
    feature_names = getattr(model, "feature_names_in_", None)
    features = _load_feature_matrix(df, feature_names)
    if features.empty:
        raise ValueError("No numeric features available to compute SHAP values.")

    sample_size = min(len(features), 500)
    sample = features.sample(sample_size, random_state=42) if len(features) > sample_size else features

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        # For classifiers, use the contribution for the positive class if available.
        shap_values = shap_values[min(1, len(shap_values) - 1)]

    shap.summary_plot(shap_values, sample, feature_names=sample.columns, show=False)
    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"✅ SHAP summary saved to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
