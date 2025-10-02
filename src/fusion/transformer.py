"""Transformer-based multimodal fusion with cross-attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import pandas as pd

from .base import BaseFusion

try:  # pragma: no cover - torch is an optional dependency
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = object  # type: ignore


@dataclass
class _ModelConfig:
    tab_dim: int
    text_dim: int


if torch is not None:

    class _CrossModalLayer(nn.Module):
        def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            ff_hidden_dim: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.tab_norm = nn.LayerNorm(embed_dim)
            self.text_norm = nn.LayerNorm(embed_dim)

            self.tab_to_text_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=False,
            )
            self.text_to_tab_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=False,
            )

            self.tab_ffn = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, ff_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_hidden_dim, embed_dim),
            )
            self.text_ffn = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, ff_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_hidden_dim, embed_dim),
            )

        def forward(
            self, tab: torch.Tensor, text: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            tab_norm = self.tab_norm(tab)
            text_norm = self.text_norm(text)

            tab_query = tab_norm.unsqueeze(0)
            text_context = text_norm.unsqueeze(0)
            attn_tab, _ = self.tab_to_text_attn(
                tab_query, text_context, text_context, need_weights=False
            )
            tab = tab + self.dropout(attn_tab.squeeze(0))

            text_query = text_norm.unsqueeze(0)
            tab_context = tab_norm.unsqueeze(0)
            attn_text, _ = self.text_to_tab_attn(
                text_query, tab_context, tab_context, need_weights=False
            )
            text = text + self.dropout(attn_text.squeeze(0))

            tab = tab + self.tab_ffn(tab)
            text = text + self.text_ffn(text)
            return tab, text

    class _TransformerFusionEncoder(nn.Module):
        def __init__(
            self,
            tab_dim: int,
            text_dim: int,
            embed_dim: int,
            num_layers: int,
            num_heads: int,
            ff_hidden_dim: int,
            dropout: float,
            fusion_dim: Optional[int],
        ) -> None:
            super().__init__()
            self.tab_embed = nn.Linear(tab_dim, embed_dim)
            self.text_embed = nn.Linear(text_dim, embed_dim)
            self.layers = nn.ModuleList(
                [
                    _CrossModalLayer(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        ff_hidden_dim=ff_hidden_dim,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.final_tab_norm = nn.LayerNorm(embed_dim)
            self.final_text_norm = nn.LayerNorm(embed_dim)
            self.out_proj = nn.Linear(embed_dim * 2, fusion_dim or embed_dim)

        def forward(self, tab: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
            tab_embed = self.tab_embed(tab)
            text_embed = self.text_embed(text)

            for layer in self.layers:
                tab_embed, text_embed = layer(tab_embed, text_embed)

            tab_embed = self.final_tab_norm(tab_embed)
            text_embed = self.final_text_norm(text_embed)
            fused = torch.cat([tab_embed, text_embed], dim=-1)
            return self.out_proj(fused)

else:  # pragma: no cover - torch unavailable fallback
    _TransformerFusionEncoder = None  # type: ignore


class TransformerFusion(BaseFusion):
    """Fuse tabular indicators with text embeddings via transformer cross-attention."""

    def __init__(
        self,
        key_columns: tuple[str, str] = ("Symbol", "Date"),
        device: Optional[str] = None,
        embed_dim: int = 128,
        fusion_dim: Optional[int] = None,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(key_columns=key_columns)
        self.device = device
        self.embed_dim = embed_dim
        self.fusion_dim = fusion_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_hidden_dim = ff_hidden_dim
        self.dropout = dropout

        self._model: Optional[Any] = None
        self._model_config: Optional[_ModelConfig] = None

    def _require_torch(self) -> None:
        if torch is None:
            raise ImportError(
                "PyTorch is required for TransformerFusion. Install torch to enable transformer-based fusion."
            )

    def _initialise_model(self, tab_dim: int, text_dim: int) -> None:
        self._require_torch()
        config = _ModelConfig(tab_dim=tab_dim, text_dim=text_dim)
        if self._model is not None and self._model_config == config:
            return

        encoder = _TransformerFusionEncoder(
            tab_dim=tab_dim,
            text_dim=text_dim,
            embed_dim=self.embed_dim,
            fusion_dim=self.fusion_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            ff_hidden_dim=self.ff_hidden_dim,
            dropout=self.dropout,
        )

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = encoder.to(device)
        self._model.eval()
        self._model_config = config

    def fuse(
        self,
        tabular_df: pd.DataFrame,
        text_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        aligned_text = self._align_text_frame(tabular_df, text_df)
        if aligned_text is None or aligned_text.empty:
            return tabular_df.loc[:, self.key_columns].copy()

        tab_numeric_cols = list(
            self._numeric_columns(tabular_df, exclude=self.key_columns)
        )
        text_numeric_cols = list(
            self._numeric_columns(aligned_text, exclude=self.key_columns)
        )

        if not tab_numeric_cols or not text_numeric_cols:
            return tabular_df.loc[:, self.key_columns].copy()

        keys = list(self.key_columns)
        tab_sorted = tabular_df.loc[:, keys + tab_numeric_cols].copy()
        tab_sorted = tab_sorted.sort_values(keys).reset_index(drop=True)
        aligned_text = aligned_text.sort_values(keys).reset_index(drop=True)

        tab_features = tab_sorted.loc[:, tab_numeric_cols].fillna(0.0)
        text_features = aligned_text.loc[:, text_numeric_cols].fillna(0.0)

        self._initialise_model(
            tab_dim=len(tab_numeric_cols), text_dim=len(text_numeric_cols)
        )
        assert self._model is not None  # for type checkers
        assert torch is not None

        device = next(self._model.parameters()).device
        tab_tensor = torch.as_tensor(
            tab_features.to_numpy(dtype="float32"), device=device
        )
        text_tensor = torch.as_tensor(
            text_features.to_numpy(dtype="float32"), device=device
        )

        with torch.no_grad():
            fused_tensor = self._model(tab_tensor, text_tensor)

        fused_numpy = fused_tensor.detach().cpu().numpy()
        feature_dim = fused_numpy.shape[1]
        column_names = [f"fusion_transformer_{idx}" for idx in range(feature_dim)]

        keys_df = tab_sorted.loc[:, self.key_columns]
        fused_features = pd.DataFrame(fused_numpy, columns=column_names)
        return pd.concat([keys_df, fused_features], axis=1)
