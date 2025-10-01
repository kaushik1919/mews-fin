import numpy as np
import pandas as pd
import pytest

from src.graph_models import GNNRiskPredictor
from src.multimodal_fusion import FusionInputs, MultiModalFeatureFusion

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


def _synthetic_graph_frame(num_days: int = 8, num_symbols: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2024-01-01", periods=num_days, freq="D")
    symbols = [f"SYM{i}" for i in range(num_symbols)]

    records = []
    for date in dates:
        for symbol in symbols:
            returns = rng.normal(scale=0.02)
            volatility = abs(returns) * 10 + rng.normal(scale=0.01)
            risk_label = int(returns < -0.01)
            records.append(
                {
                    "Date": date,
                    "Symbol": symbol,
                    "Returns": returns,
                    "Volatility_20d": volatility,
                    "Risk_Label": risk_label,
                    "Risk_Score": volatility,
                }
            )
    return pd.DataFrame(records)


def test_gnn_risk_predictor_generates_features() -> None:
    frame = _synthetic_graph_frame()
    predictor = GNNRiskPredictor(
        window=5,
        correlation_threshold=0.1,
        epochs=5,
        hidden_channels=16,
        learning_rate=5e-3,
        patience=3,
        random_state=7,
    )

    features = predictor.generate_features(frame)

    assert not features.empty
    assert {"Symbol", "Date", "gnn_risk_score"}.issubset(features.columns)
    assert features["gnn_risk_score"].between(-5, 5).all()


def test_multimodal_fusion_adds_gnn_features() -> None:
    base_df = _synthetic_graph_frame(num_days=6)
    fusion = MultiModalFeatureFusion(
        fusion_strategy="concat",
        enable_gnn=True,
        gnn_kwargs={
            "window": 3,
            "epochs": 4,
            "correlation_threshold": 0.05,
            "hidden_channels": 8,
            "learning_rate": 1e-2,
            "patience": 2,
            "random_state": 21,
        },
    )

    fused = fusion.fuse(
        FusionInputs(
            tabular_features=base_df,
            graph_source=base_df,
        )
    )

    gnn_cols = [col for col in fused.columns if col.startswith("gnn_risk_")]
    assert gnn_cols, "Expected GNN risk features to be present in fused dataframe"
    assert fused[gnn_cols].notna().any().any()
