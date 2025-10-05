"""Cross-modal fusion utilities leveraging attention mechanisms."""

from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - torch optional dependency
    torch = None  # type: ignore
    nn = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    import torch as _torch


class CrossAttentionFusion(nn.Module):
    """Fuse tabular and textual representations using cross-attention."""

    def __init__(
        self,
        tabular_dim: int,
        text_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        if torch is None or nn is None:
            raise ImportError("PyTorch is required for CrossAttentionFusion")

        super().__init__()
        self.tabular_proj = nn.Linear(tabular_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm_tab = nn.LayerNorm(hidden_dim)
        self.norm_text = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        tabular: "_torch.Tensor",
        text: "_torch.Tensor",
    ) -> "_torch.Tensor":
        if tabular.dim() == 1:
            tabular = tabular.unsqueeze(0)
        if text.dim() == 1:
            text = text.unsqueeze(0)

        tab_emb = self.tabular_proj(tabular)
        text_emb = self.text_proj(text)

        attn_output, _ = self.cross_attn(
            query=tab_emb.unsqueeze(0),
            key=text_emb.unsqueeze(0),
            value=text_emb.unsqueeze(0),
        )
        attn_output = attn_output.squeeze(0)
        tab_enhanced = self.norm_tab(tab_emb + attn_output)

        reverse_attn, _ = self.cross_attn(
            query=text_emb.unsqueeze(0),
            key=tab_emb.unsqueeze(0),
            value=tab_emb.unsqueeze(0),
        )
        reverse_attn = reverse_attn.squeeze(0)
        text_enhanced = self.norm_text(text_emb + reverse_attn)

        fused = torch.cat([tab_enhanced, text_enhanced], dim=-1)
        fused = self.output_proj(self.ffn(fused))
        return fused

    def fuse_dataframes(
        self,
        tabular_df: pd.DataFrame,
        text_df: pd.DataFrame,
        key_columns: Tuple[str, str] = ("Symbol", "Date"),
        device: Optional[str] = None,
    ) -> pd.DataFrame:
        if torch is None:
            raise ImportError("PyTorch is required for CrossAttentionFusion")

        symbol_col, date_col = key_columns
        date_col = date_col

        joined = tabular_df.merge(text_df, on=[symbol_col, date_col], how="left")
        joined.sort_values([symbol_col, date_col], inplace=True)

        tab_cols = [col for col in tabular_df.columns if col not in key_columns]
        text_cols = [col for col in text_df.columns if col not in key_columns]

        fused_rows = []
        used_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(used_device)
        self.eval()

        for _, row in joined.iterrows():
            tab_vector = torch.tensor(
                row[tab_cols].fillna(0.0).values,
                dtype=torch.float32,
                device=used_device,
            )
            text_vector = torch.tensor(
                row[text_cols].fillna(0.0).values,
                dtype=torch.float32,
                device=used_device,
            )
            with torch.no_grad():
                fused = self.forward(tab_vector, text_vector)
            fused_rows.append(fused.cpu().numpy())

        fused_array = np.vstack(fused_rows)
        fused_df = pd.DataFrame(
            fused_array,
            columns=[f"fusion_feature_{i}" for i in range(fused_array.shape[1])],
        )
        fused_df[symbol_col] = joined[symbol_col].values
        fused_df[date_col] = joined[date_col].values
        return fused_df


__all__ = ["CrossAttentionFusion"]
