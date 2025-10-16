"""Multi-modal feature fusion module for MEWS.

This module combines tabular financial indicators, news embeddings, and
graph-based features into a unified feature matrix ready for downstream
machine learning models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:  # Optional dependency for graph analytics
    import networkx as nx
except ImportError:  # pragma: no cover - networkx is optional
    nx = None  # type: ignore

try:  # Optional dependency for transformer embeddings
    import torch
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover - transformers is optional
    torch = None  # type: ignore
    AutoModel = None  # type: ignore
    AutoTokenizer = None  # type: ignore

try:  # Optional dependency for GNN-based risk features
    from src.graph_models import GNNRiskPredictor
except ImportError:  # pragma: no cover - torch geometric optional
    GNNRiskPredictor = None  # type: ignore

from src.config import Config
from src.fusion import BaseFusion
from src.fusion import CrossAttentionFusion as FusionCrossAttention
from src.fusion import GatedFusion, SimpleConcatFusion
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class FusionInputs:
    """Container for multimodal data sources."""

    tabular_features: pd.DataFrame
    news_df: Optional[pd.DataFrame] = None
    graph_source: Optional[pd.DataFrame] = None
    text_column: str = "headline"
    news_symbol_column: str = "Symbol"
    news_datetime_column: str = "Date"


class MultiModalFeatureFusion:
    """Fuse tabular, news, and graph features into a single matrix.

    The class lazily loads optional transformer models if available and provides
    guardrails so the MEWS pipeline can operate even when optional dependencies
    are not installed. All methods are pure functions over pandas data frames to
    simplify integration with existing preprocessing steps.
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        fusion_strategy: str = "concat",
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        enable_gnn: bool = True,
        gnn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.device = device or (
            "cuda" if torch and torch.cuda.is_available() else "cpu"
        )
        self._tokenizer = None
        self._model = None
        self.embedding_dim = None
        self.cross_attention_kwargs = cross_attention_kwargs or {}
        self.fusion_strategy = "concat"
        self._fusion_impl = self._build_fusion_strategy(fusion_strategy)
        self.enable_gnn = enable_gnn
        self.gnn_kwargs = gnn_kwargs or {}
        self._gnn_predictor = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def fuse(self, inputs: FusionInputs) -> pd.DataFrame:
        """Combine multimodal signals into a training-ready feature set.

        Args:
            inputs: Collection of tabular, news, and graph data.

        Returns:
            DataFrame aligned on ["Symbol", "Date"] with fused features.
        """

        base_df = inputs.tabular_features.copy()
        base_df = self._ensure_datetime(base_df)

        if inputs.news_df is not None and not inputs.news_df.empty:
            news_features = self._compute_news_embeddings(
                inputs.news_df,
                text_column=inputs.text_column,
                symbol_column=inputs.news_symbol_column,
                datetime_column=inputs.news_datetime_column,
            )

            fusion_features: Optional[pd.DataFrame] = None
            if news_features is not None and not news_features.empty:
                try:
                    fusion_features = self._fusion_impl.fuse(base_df, news_features)
                except ImportError as exc:
                    LOGGER.warning(
                        "Fusion strategy '%s' unavailable (%s); falling back to simple concatenation.",
                        self.fusion_strategy,
                        exc,
                    )
                    self.fusion_strategy = "concat"
                    self._fusion_impl = SimpleConcatFusion()
                    fusion_features = self._fusion_impl.fuse(base_df, news_features)
                except Exception as exc:  # pragma: no cover - defensive guard
                    LOGGER.warning(
                        "Fusion strategy '%s' failed with error: %s",
                        self.fusion_strategy,
                        exc,
                    )
                    if self.fusion_strategy != "concat":
                        self.fusion_strategy = "concat"
                        self._fusion_impl = SimpleConcatFusion()
                        fusion_features = self._fusion_impl.fuse(base_df, news_features)

            if fusion_features is not None:
                feature_cols = [
                    col
                    for col in fusion_features.columns
                    if col not in self._fusion_impl.key_columns
                ]
                if feature_cols:
                    base_df = self._merge_features(base_df, fusion_features)
                else:
                    base_df = self._merge_features(base_df, news_features)
            elif news_features is not None and not news_features.empty:
                base_df = self._merge_features(base_df, news_features)

        graph_features = self._build_graph_features(
            base_df if inputs.graph_source is None else inputs.graph_source
        )
        if graph_features is not None and not graph_features.empty:
            base_df = self._merge_features(base_df, graph_features)

        base_df = base_df.sort_values(["Symbol", "Date"])
        base_df = base_df.reset_index(drop=True)
        return base_df

    # ------------------------------------------------------------------
    # Strategy configuration
    # ------------------------------------------------------------------
    def _build_fusion_strategy(self, strategy: str) -> BaseFusion:
        strategy_key = (strategy or "concat").lower()
        self.fusion_strategy = strategy_key

        if strategy_key == "concat":
            return SimpleConcatFusion()
        if strategy_key == "cross_attention":
            return FusionCrossAttention(
                key_columns=("Symbol", "Date"),
                device=self.device,
                **self.cross_attention_kwargs,
            )
        if strategy_key == "gated":
            return GatedFusion()

        LOGGER.warning(
            "Unknown fusion strategy '%s'; defaulting to simple concatenation.",
            strategy,
        )
        self.fusion_strategy = "concat"
        return SimpleConcatFusion()

    # ------------------------------------------------------------------
    # News embeddings
    # ------------------------------------------------------------------
    def _compute_news_embeddings(
        self,
        news_df: pd.DataFrame,
        text_column: str,
        symbol_column: str,
        datetime_column: str,
    ) -> pd.DataFrame:
        """Aggregate transformer embeddings of news text by symbol and date."""

        news_clean = news_df.copy()
        missing_cols = {
            text_column,
            symbol_column,
            datetime_column,
        } - set(news_clean.columns)
        if missing_cols:
            LOGGER.warning("News dataframe missing columns: %s", missing_cols)
            return pd.DataFrame()

        news_clean[datetime_column] = pd.to_datetime(news_clean[datetime_column])
        news_clean["Date"] = news_clean[datetime_column].dt.floor("D")
        news_clean.rename(columns={symbol_column: "Symbol"}, inplace=True)

        if AutoTokenizer is None or AutoModel is None or torch is None:
            LOGGER.warning(
                "Transformers not available, returning sparse news features."
            )
            aggregated = (
                news_clean.groupby(["Symbol", "Date"])[text_column]
                .count()
                .rename("news_article_count")
                .reset_index()
            )
            return aggregated

        if self._tokenizer is None or self._model is None:
            self._load_embedding_model()

        embeddings = []
        for text in news_clean[text_column].fillna(""):
            if not text.strip():
                embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
                continue

            inputs = self._tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self._model(**inputs.to(self.device))  # type: ignore[attr-defined]
                sentence_embedding = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(sentence_embedding.cpu().numpy()[0])

        embedding_matrix = np.vstack(embeddings)
        embedding_cols = [
            f"news_embedding_{i}" for i in range(embedding_matrix.shape[1])
        ]
        news_clean = news_clean.assign(
            **{col: embedding_matrix[:, idx] for idx, col in enumerate(embedding_cols)}
        )

        aggregated = (
            news_clean.groupby(["Symbol", "Date"])[embedding_cols].mean().reset_index()
        )
        aggregated["news_article_count"] = (
            news_clean.groupby(["Symbol", "Date"])[text_column].count().values
        )
        return aggregated

    def _load_embedding_model(self) -> None:
        """Lazy-load the transformer model for embeddings."""

        if AutoTokenizer is None or AutoModel is None or torch is None:
            raise RuntimeError("transformers library is required for embeddings")

        LOGGER.info("Loading embedding model %s", self.embedding_model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.embedding_model_name, revision=Config.HF_MODEL_REVISION
        )
        self._model = AutoModel.from_pretrained(
            self.embedding_model_name, revision=Config.HF_MODEL_REVISION
        )
        self._model.to(self.device)  # type: ignore[union-attr]
        self._model.eval()
        self.embedding_dim = int(self._model.config.hidden_size)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Graph/network features
    # ------------------------------------------------------------------
    def _build_graph_features(
        self, df: pd.DataFrame, window: int = 60
    ) -> Optional[pd.DataFrame]:
        centrality_df: Optional[pd.DataFrame] = None
        if nx is not None:
            if not {"Symbol", "Date", "Returns"}.issubset(df.columns):
                LOGGER.warning("Insufficient columns for graph features")
            else:
                df_sorted = df.sort_values(["Date", "Symbol"]).copy()
                df_sorted["Date"] = pd.to_datetime(df_sorted["Date"])

                returns_wide = df_sorted.pivot_table(
                    index="Date", columns="Symbol", values="Returns", aggfunc="mean"
                ).sort_index()

                feature_rows = []
                for date in returns_wide.index:
                    window_slice = returns_wide.loc[:date].tail(window)
                    if (
                        window_slice.shape[0] < max(10, window // 2)
                        or window_slice.shape[1] < 3
                    ):
                        continue
                    corr_matrix = window_slice.corr().fillna(0)
                    graph = nx.from_pandas_adjacency(corr_matrix)
                    try:
                        centrality = nx.eigenvector_centrality_numpy(
                            graph, weight="weight"
                        )
                    except Exception:  # pragma: no cover - numerical issues
                        centrality = nx.degree_centrality(graph)
                    clustering = nx.clustering(graph, weight="weight")

                    for symbol in corr_matrix.columns:
                        feature_rows.append(
                            {
                                "Symbol": symbol,
                                "Date": date,
                                "graph_eigenvector": centrality.get(symbol, 0.0),
                                "graph_clustering": clustering.get(symbol, 0.0),
                            }
                        )

                if feature_rows:
                    centrality_df = pd.DataFrame(feature_rows)
                    centrality_df.sort_values(["Symbol", "Date"], inplace=True)
        else:
            LOGGER.warning("networkx not available, skipping graph centrality features")

        gnn_features: Optional[pd.DataFrame] = None
        if self.enable_gnn:
            if GNNRiskPredictor is None:
                LOGGER.warning(
                    "torch-geometric not available, skipping GNN risk features"
                )
            else:
                try:
                    predictor = self._get_gnn_predictor()
                    gnn_features = predictor.generate_features(df)
                except ImportError as exc:
                    LOGGER.warning("GNN risk predictor unavailable (%s)", exc)
                except Exception as exc:  # pragma: no cover - defensive guard
                    LOGGER.warning("GNN risk predictor failed with error: %s", exc)

        frames = [
            frame
            for frame in (centrality_df, gnn_features)
            if frame is not None and not frame.empty
        ]
        if not frames:
            return None

        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on=["Symbol", "Date"], how="outer")

        merged.sort_values(["Symbol", "Date"], inplace=True)
        return merged

    def _get_gnn_predictor(self):
        if not self.enable_gnn:
            raise RuntimeError("GNN predictor requested while disabled")
        if self._gnn_predictor is None:
            if GNNRiskPredictor is None:
                raise ImportError("torch-geometric is required for GNN risk features")
            self._gnn_predictor = GNNRiskPredictor(**self.gnn_kwargs)
        return self._gnn_predictor

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _merge_features(
        self, base_df: pd.DataFrame, features_df: pd.DataFrame
    ) -> pd.DataFrame:
        merge_cols = [col for col in ["Symbol", "Date"] if col in features_df.columns]
        if not merge_cols:
            LOGGER.warning(
                "Feature dataframe missing merge keys: %s", features_df.columns
            )
            return base_df

        merged = base_df.merge(features_df, on=merge_cols, how="left")
        merged.sort_values(["Symbol", "Date"], inplace=True)
        feature_cols = [col for col in features_df.columns if col not in merge_cols]
        if feature_cols:
            merged[feature_cols] = merged.groupby("Symbol")[feature_cols].ffill()
        return merged

    @staticmethod
    def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
        if "Date" in df.columns:
            df = df.copy()
            df["Date"] = pd.to_datetime(df["Date"])
        return df


__all__ = ["FusionInputs", "MultiModalFeatureFusion"]
