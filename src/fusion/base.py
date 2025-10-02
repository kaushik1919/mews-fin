"""Base classes for multimodal fusion strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd


class BaseFusion(ABC):
    """Abstract interface for combining tabular and auxiliary modalities."""

    def __init__(self, key_columns: Tuple[str, str] = ("Symbol", "Date")) -> None:
        self.key_columns = key_columns

    def _ensure_keys_present(self, df: pd.DataFrame, name: str) -> None:
        missing = [col for col in self.key_columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame '{name}' is missing required key column(s): {missing}"
            )

    def _numeric_columns(
        self, df: pd.DataFrame, exclude: Optional[Sequence[str]] = None
    ) -> Sequence[str]:
        exclude = exclude or ()
        return [
            col
            for col in df.columns
            if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
        ]

    def _align_text_frame(
        self,
        tabular_df: pd.DataFrame,
        text_df: Optional[pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        if text_df is None or text_df.empty:
            return None

        self._ensure_keys_present(tabular_df, "tabular_df")
        self._ensure_keys_present(text_df, "text_df")

        key_df = tabular_df.loc[:, self.key_columns].copy()
        aligned = key_df.merge(text_df, on=self.key_columns, how="left")
        aligned.sort_values(list(self.key_columns), inplace=True)
        aligned.reset_index(drop=True, inplace=True)
        numeric_cols = self._numeric_columns(aligned, exclude=self.key_columns)
        if numeric_cols:
            aligned[numeric_cols] = aligned[numeric_cols].fillna(0.0)
        return aligned

    @abstractmethod
    def fuse(
        self,
        tabular_df: pd.DataFrame,
        text_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Return fused features aligned on the configured key columns."""

    def empty_result(self, rows: int) -> pd.DataFrame:
        """Utility to create an empty result with key columns for merging."""

        frame = pd.DataFrame({col: [None] * rows for col in self.key_columns})
        return frame
