"""Tests for robustness utilities and evaluators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.robustness import (
    AdversarialNoiseTester,
    DelaySimulator,
    PerturbationConfig,
    RobustnessEvaluator,
    SentimentBiasAuditor,
)


def test_sentiment_bias_auditor_reports_difference():
    df = pd.DataFrame(
        {
            "Symbol": ["AAA", "AAA", "BBB", "BBB"],
            "finbert_sentiment": [0.2, 0.3, 0.4, 0.5],
            "vader_compound": [-0.1, 0.0, 0.1, 0.0],
        }
    )
    auditor = SentimentBiasAuditor(
        finbert_columns=["finbert_sentiment"],
        vader_columns=["vader_compound"],
    )
    report = auditor.audit(df, group_col="Symbol")

    assert report["comparisons"]
    comparison = report["comparisons"][0]
    assert pytest.approx(comparison.mean_difference, rel=1e-6) == pytest.approx(
        df["finbert_sentiment"].mean() - df["vader_compound"].mean(), rel=1e-6
    )
    assert report["group_summaries"]


def test_adversarial_noise_tester_applies_noise():
    df = pd.DataFrame({"feature": np.linspace(0.0, 1.0, 10)})
    tester = AdversarialNoiseTester()
    perturbed, report = tester.apply(df, noise_level=0.5, random_state=1)

    assert not np.allclose(perturbed["feature"], df["feature"])
    assert report.noise_level == 0.5
    assert "feature" in report.affected_columns


def test_delay_simulator_shifts_sentiment():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "Symbol": ["AAA"] * 4,
            "news_sentiment": [0.1, 0.2, 0.3, 0.4],
        }
    )
    simulator = DelaySimulator(date_col="Date")
    perturbed, report = simulator.apply(df, delay_days=1)

    assert pytest.approx(perturbed.loc[1, "news_sentiment"]) == pytest.approx(0.1)
    assert report.delay_days == 1


def test_robustness_evaluator_runs_with_stub(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "Symbol": ["AAA"] * 6,
            "feature": np.linspace(0.0, 1.0, 6),
            "Risk_Label": [0, 0, 0, 1, 0, 1],
        }
    )

    class DummyPredictor:
        def __init__(self) -> None:
            self.training_metadata = None
            self.models = {}
            self.thresholds = {}
            self.ensemble_threshold = 0.5
            self.ensemble_weights = None
            self.dynamic_ensemble = None
            self.regime_detector = None

        def prepare_modeling_data(
            self, df, feature_groups=None, target_col="Risk_Label"
        ):
            features = df[["feature"]].copy()
            target = df[target_col].to_numpy()
            self.training_metadata = df[["Date", "Symbol"]].copy()
            return features, target, ["feature"]

        def train_models(self, X, y, feature_names, test_size=0.2, random_state=42):
            self.models = {"random_forest": object()}
            return {
                "ensemble": {
                    "auc_score": 0.7,
                    "test_accuracy": 0.6,
                    "fbeta_score": 0.5,
                }
            }

        def predict_risk(self, X, model_type="ensemble", metadata=None):
            probabilities = np.full(len(X), 0.6)
            predictions = (probabilities >= self.ensemble_threshold).astype(int)
            return predictions, probabilities

    monkeypatch.setattr("src.robustness.evaluator.RiskPredictor", DummyPredictor)

    evaluator = RobustnessEvaluator(dataset=df, output_root=tmp_path)
    report = evaluator.run(
        perturbations=[
            PerturbationConfig(
                name="noise_zero", kind="noise", params={"noise_level": 0.0}
            ),
            PerturbationConfig(
                name="delay_zero", kind="delay", params={"delay_days": 0}
            ),
        ],
        auditor=SentimentBiasAuditor(
            finbert_columns=["feature"],
            vader_columns=["feature"],
        ),
    )

    assert report.baseline_metrics["ensemble"]["auc_score"] == 0.7
    assert len(report.perturbations) == 2
    for outcome in report.perturbations:
        assert "auc_score" in outcome.metrics["ensemble"]
        assert outcome.deltas["auc_score"] == pytest.approx(0.0)

    output_files = list(tmp_path.glob("robustness_*/robustness_report.json"))
    assert output_files
