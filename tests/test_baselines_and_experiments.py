"""Tests for baseline models and experiment manager."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from src.baselines import GARCHBaseline, LSTMBaseline, ValueAtRiskBaseline
from src.experiments import ExperimentConfig, ExperimentManager


def _make_sample_frame(rows: int = 200, symbols: int = 2) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    frames = []
    rng = np.random.default_rng(42)
    for idx in range(symbols):
        returns = rng.normal(loc=0.0, scale=0.02, size=rows)
        labels = (rng.random(rows) > 0.7).astype(int)
        frames.append(
            pd.DataFrame(
                {
                    "Date": dates,
                    "Symbol": f"SYM{idx}",
                    "Returns": returns,
                    "Risk_Label": labels,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_value_at_risk_baseline_thresholds():
    df = _make_sample_frame(rows=150, symbols=1)
    baseline = ValueAtRiskBaseline(confidence=0.9, window=30)
    result = baseline.run(df)

    assert not result.predictions.empty
    assert {"var_hist_threshold", "var_param_threshold"}.issubset(result.predictions.columns)
    assert result.metadata["confidence_level"] == 0.9


def test_lstm_baseline_generates_probabilities(tmp_path):
    pytest.importorskip("torch")
    df = _make_sample_frame(rows=80, symbols=1)
    baseline = LSTMBaseline(
        sequence_length=5,
        hidden_size=8,
        epochs=1,
        batch_size=16,
    )
    result = baseline.run(df)
    assert "lstm_risk_probability" in result.predictions.columns
    assert result.predictions["lstm_risk_probability"].between(0.0, 1.0).all()


def test_garch_baseline_runs_when_arch_available():
    pytest.importorskip("arch")
    df = _make_sample_frame(rows=160, symbols=1)
    baseline = GARCHBaseline(confidence=0.9, min_observations=120)
    result = baseline.run(df)
    assert "garch_var_threshold" in result.predictions.columns
    assert not result.predictions["garch_var_threshold"].isna().all()


def test_experiment_manager_executes(tmp_path):
    df = _make_sample_frame(rows=120, symbols=1)
    config = ExperimentConfig(
        name="unit_test",
        baselines=[{"type": "value_at_risk", "params": {"confidence": 0.9, "window": 20}}],
        mews={"enabled": False},
        output_dir=str(tmp_path),
    )
    manager = ExperimentManager(config=config, dataframe=df)
    summary = manager.run()

    assert summary["experiment"] == "unit_test"
    assert summary["runs"]
    combined_files = [run.get("combined_baselines") for run in summary["runs"] if run.get("combined_baselines")]
    assert combined_files
    for file_path in combined_files:
        assert Path(file_path).exists()


def test_experiment_regime_integration(tmp_path):
    rng = np.random.default_rng(7)
    df = _make_sample_frame(rows=180, symbols=1)
    df["Returns"] = rng.normal(0, 0.02, size=len(df))

    config = ExperimentConfig(
        name="regime_test",
        baselines=[],
        mews={
            "enabled": True,
            "test_size": 0.3,
            "regime_adaptive": {
                "enabled": True,
                "lookback": 14,
                "use_meta_model": False,
            },
        },
        output_dir=str(tmp_path),
    )
    manager = ExperimentManager(config=config, dataframe=df)
    summary = manager.run()

    mews_payload = summary["runs"][0].get("mews_model")
    assert mews_payload is not None
    regime_payload = mews_payload.get("regime_adaptive")
    assert regime_payload is not None
    artifact_path = regime_payload.get("artifact")
    assert artifact_path and Path(artifact_path).exists()

    with Path(artifact_path).open("r", encoding="utf-8") as handle:
        regime_summary = json.load(handle)
    assert "regimes" in regime_summary
    assert regime_payload["config"]["lookback"] == 14
