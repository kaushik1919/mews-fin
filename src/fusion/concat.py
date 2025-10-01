"""Simple concatenation fusion strategy."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .base import BaseFusion


class SimpleConcatFusion(BaseFusion):
    """Baseline fusion that concatenates tabular and text-derived features."""

    def fuse(
        self,
        tabular_df: pd.DataFrame,
        text_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        aligned_text = self._align_text_frame(tabular_df, text_df)
        if aligned_text is None:
            return tabular_df.loc[:, self.key_columns].copy()

        # Drop any duplicate columns from text that already exist in tabular data.
        duplicate_cols = {
            col for col in aligned_text.columns if col not in self.key_columns and col in tabular_df.columns
        }
        if duplicate_cols:
            aligned_text = aligned_text.drop(columns=list(duplicate_cols))

        return aligned_text
