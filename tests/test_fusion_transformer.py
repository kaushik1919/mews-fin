import numpy as np
import pandas as pd
import pytest

from src.fusion.transformer import TransformerFusion

torch = pytest.importorskip("torch")


def _sample_frames(rows: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    symbols = ["AAA"] * rows

    tabular_df = pd.DataFrame(
        {
            "Symbol": symbols,
            "Date": dates,
            "indicator_0": rng.normal(size=rows),
            "indicator_1": rng.normal(size=rows),
            "indicator_2": rng.normal(size=rows),
        }
    )

    text_df = pd.DataFrame(
        {
            "Symbol": symbols,
            "Date": dates,
            "finbert_0": rng.normal(size=rows),
            "finbert_1": rng.normal(size=rows),
            "vader_pos": rng.normal(size=rows),
            "vader_neg": rng.normal(size=rows),
        }
    )
    return tabular_df, text_df


def test_transformer_fusion_produces_fused_features() -> None:
    tabular_df, text_df = _sample_frames()
    fusion = TransformerFusion(
        embed_dim=16,
        fusion_dim=8,
        num_layers=2,
        num_heads=2,
        ff_hidden_dim=32,
        dropout=0.0,
        device="cpu",
    )

    result = fusion.fuse(tabular_df, text_df)

    assert list(result.columns[:2]) == ["Symbol", "Date"]
    assert result.shape == (len(tabular_df), 10)
    assert not result.iloc[:, 2:].isna().any().any()


def test_transformer_fusion_gracefully_handles_missing_text() -> None:
    tabular_df, _ = _sample_frames()
    fusion = TransformerFusion()

    result = fusion.fuse(tabular_df, None)

    assert list(result.columns) == ["Symbol", "Date"]
    assert len(result) == len(tabular_df)
