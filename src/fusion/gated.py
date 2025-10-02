"""Gated fusion strategy balancing tabular and text modalities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseFusion


class GatedFusion(BaseFusion):
    """Applies a soft gating mechanism to blend tabular and text features."""

    def __init__(self, key_columns: tuple[str, str] = ("Symbol", "Date")) -> None:
        super().__init__(key_columns=key_columns)

    def fuse(
        self,
        tabular_df: pd.DataFrame,
        text_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        aligned_text = self._align_text_frame(tabular_df, text_df)
        rows = len(tabular_df)
        if aligned_text is None:
            result = tabular_df.loc[:, self.key_columns].copy()
            result["fusion_gate"] = np.ones(rows, dtype=float)
            return result

        tab_numeric_cols = self._numeric_columns(tabular_df, exclude=self.key_columns)
        text_numeric_cols = self._numeric_columns(
            aligned_text, exclude=self.key_columns
        )
        if not tab_numeric_cols:
            return aligned_text

        tab_numeric = (
            tabular_df.loc[:, tab_numeric_cols].fillna(0.0).to_numpy(dtype=float)
        )
        text_numeric = (
            aligned_text.loc[:, text_numeric_cols].fillna(0.0).to_numpy(dtype=float)
            if text_numeric_cols
            else np.zeros((rows, 1), dtype=float)
        )

        tab_norm = np.linalg.norm(tab_numeric, axis=1, keepdims=True)
        text_norm = np.linalg.norm(text_numeric, axis=1, keepdims=True)
        gate = tab_norm / (tab_norm + text_norm + 1e-6)
        gate = np.clip(gate, 0.0, 1.0)

        gated_tab = tab_numeric * gate
        gated_text = text_numeric * (1.0 - gate)

        result = tabular_df.loc[:, self.key_columns].copy()
        result["fusion_gate"] = gate.squeeze()

        gated_tab_df = pd.DataFrame(
            gated_tab,
            columns=[f"gated_tab_{col}" for col in tab_numeric_cols],
        )
        gated_text_df = pd.DataFrame(
            gated_text,
            columns=(
                [f"gated_text_{col}" for col in text_numeric_cols]
                if text_numeric_cols
                else ["gated_text_0"]
            ),
        )

        merged = pd.concat(
            [result.reset_index(drop=True), gated_tab_df, gated_text_df], axis=1
        )
        return merged
