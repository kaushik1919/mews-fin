"""Cross-attention fusion strategy leveraging transformer-style attention."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .base import BaseFusion

try:  # Optional dependency via research module
    from src.research.cross_modal import CrossAttentionFusion as _CrossAttentionModule
except ImportError:  # pragma: no cover - torch optional
    _CrossAttentionModule = None  # type: ignore


class CrossAttentionFusion(BaseFusion):
    """Fuse modalities using a learned cross-attention projection."""

    def __init__(
        self,
        key_columns: tuple[str, str] = ("Symbol", "Date"),
        device: Optional[str] = None,
        include_text_features: bool = True,
        **fusion_kwargs: Any,
    ) -> None:
        super().__init__(key_columns=key_columns)
        self.device = device
        self.include_text_features = include_text_features
        self.fusion_kwargs: Dict[str, Any] = dict(fusion_kwargs)
        self._model: Optional[Any] = None

    def fuse(
        self,
        tabular_df: pd.DataFrame,
        text_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if _CrossAttentionModule is None:
            raise ImportError("PyTorch and the research cross-modal module are required for cross-attention fusion")

        aligned_text = self._align_text_frame(tabular_df, text_df)
        if aligned_text is None:
            return tabular_df.loc[:, self.key_columns].copy()

        tab_numeric_cols = self._numeric_columns(tabular_df, exclude=self.key_columns)
        text_numeric_cols = self._numeric_columns(aligned_text, exclude=self.key_columns)
        if not tab_numeric_cols or not text_numeric_cols:
            return tabular_df.loc[:, self.key_columns].copy()

        tab_subset = tabular_df.loc[:, list(self.key_columns) + list(tab_numeric_cols)].copy()
        text_subset = aligned_text.loc[:, list(self.key_columns) + list(text_numeric_cols)].copy()

        if self._model is None:
            init_kwargs: Dict[str, Any] = {
                "tabular_dim": len(tab_numeric_cols),
                "text_dim": len(text_numeric_cols),
            }
            init_kwargs.update(self.fusion_kwargs)
            self._model = _CrossAttentionModule(**init_kwargs)

        fused_df = self._model.fuse_dataframes(
            tabular_df=tab_subset,
            text_df=text_subset,
            key_columns=self.key_columns,
            device=self.device,
        )

        if not self.include_text_features:
            return fused_df

        # Merge the original text features (prefixed) with the fused attention outputs.
        text_prefixed = text_subset.copy()
        text_prefixed = text_prefixed.rename(
            columns={col: f"text_{col}" for col in text_numeric_cols}
        )
        merged = fused_df.merge(
            text_prefixed,
            on=list(self.key_columns),
            how="left",
        )
        return merged
