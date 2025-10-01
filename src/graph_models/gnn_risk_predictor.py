"""Graph neural network risk predictor for MEWS.

This module builds correlation graphs over equities and applies a lightweight
Graph Neural Network (GCN or GAT) implemented with PyTorch Geometric to
produce node-level risk scores that can be re-used as engineered features in
traditional classifiers or ensembling steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - torch is optional at runtime
    torch = None  # type: ignore
    nn = object  # type: ignore
    F = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv, GCNConv
    from torch_geometric.utils import add_self_loops
except ImportError:  # pragma: no cover - torch geometric optional
    Data = None  # type: ignore
    GATConv = None  # type: ignore
    GCNConv = None  # type: ignore
    add_self_loops = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from src.uncertainty.monte_carlo_dropout import monte_carlo_dropout
except ImportError:  # pragma: no cover
    monte_carlo_dropout = None  # type: ignore

LOGGER = get_logger(__name__)


if torch is not None and isinstance(nn, type) and Data is not None:

    class GraphRiskNet(nn.Module):
        """Tiny GNN used to approximate node-level risk."""

        def __init__(
            self,
            input_dim: int,
            hidden_channels: int = 32,
            gnn_type: str = "gcn",
            heads: int = 2,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.gnn_type = gnn_type
            self.dropout_layer = nn.Dropout(dropout)

            if gnn_type == "gat":
                self.conv1 = GATConv(
                    in_channels=input_dim,
                    out_channels=hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    concat=True,
                )
                self.conv2 = GATConv(
                    in_channels=hidden_channels * heads,
                    out_channels=hidden_channels,
                    heads=1,
                    dropout=dropout,
                    concat=False,
                )
            else:  # default to GCN
                self.conv1 = GCNConv(input_dim, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, hidden_channels)

            self.out_linear = nn.Linear(hidden_channels, 1)

        def forward(  # type: ignore[override]
            self,
            x,
            edge_index,
            edge_weight=None,
        ):
            if self.gnn_type == "gat":
                x = self.conv1(x, edge_index)
            else:
                x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.dropout_layer(x)

            if self.gnn_type == "gat":
                x = self.conv2(x, edge_index)
            else:
                x = self.conv2(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.dropout_layer(x)

            out = self.out_linear(x)
            return out.squeeze(-1)


else:  # pragma: no cover - fallback definition when torch is unavailable
    GraphRiskNet = None  # type: ignore


@dataclass
class GNNMetadata:
    symbols: List[str]
    date: pd.Timestamp
    is_classification: bool
    target_column: Optional[str]
    graph_density: float
    target_std: float


class GNNRiskPredictor:
    """Generate risk scores from correlation graphs via PyTorch Geometric."""

    def __init__(
        self,
        window: int = 60,
        correlation_threshold: float = 0.5,
        gnn_type: str = "gcn",
        hidden_channels: int = 32,
        heads: int = 2,
        dropout: float = 0.1,
        epochs: int = 80,
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4,
        target_column: str = "Risk_Label",
        fallback_target_columns: Sequence[str] = ("Risk_Score", "Volatility_20d", "Returns"),
        device: Optional[str] = None,
        patience: int = 10,
        tolerance: float = 1e-5,
        random_state: int = 42,
        mc_dropout_samples: int = 0,
    ) -> None:
        self.window = max(int(window), 10)
        self.correlation_threshold = float(correlation_threshold)
        self.hidden_channels = int(hidden_channels)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.target_column = target_column
        self.fallback_target_columns = tuple(fallback_target_columns)
        self.patience = max(int(patience), 1)
        self.tolerance = float(tolerance)
        self.random_state = int(random_state)
        self.logger = get_logger(__name__)
        self.mc_dropout_samples = int(mc_dropout_samples)

        gnn_key = gnn_type.lower()
        if gnn_key not in {"gcn", "gat"}:
            raise ValueError("gnn_type must be either 'gcn' or 'gat'")
        self.gnn_type = gnn_key

        if device:
            self.device = device
        else:
            if torch is not None and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return node-level GNN risk features for downstream models."""

        try:
            self._check_dependencies()
        except ImportError as exc:  # pragma: no cover - optional dependency path
            self.logger.warning(
                "Skipping GNN risk predictor because dependencies are missing: %s",
                exc,
            )
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        working_df = df.copy()
        if "Date" not in working_df.columns or "Symbol" not in working_df.columns:
            self.logger.warning("Dataframe must contain 'Date' and 'Symbol' columns")
            return pd.DataFrame()

        working_df["Date"] = pd.to_datetime(working_df["Date"])
        working_df.sort_values(["Date", "Symbol"], inplace=True)

        unique_dates = working_df["Date"].dropna().unique()
        if len(unique_dates) < self.window:
            self.logger.info(
                "Insufficient history (%s dates) for GNN window size %s",
                len(unique_dates),
                self.window,
            )
            return pd.DataFrame()

        feature_frames: List[pd.DataFrame] = []
        for idx in range(self.window - 1, len(unique_dates)):
            window_dates = unique_dates[idx - self.window + 1 : idx + 1]
            window_df = working_df[working_df["Date"].isin(window_dates)]
            prepared = self._prepare_graph_data(window_df)
            if prepared is None:
                continue

            data, metadata = prepared
            predictions, mc_stats = self._train_and_predict(data, metadata)

            feature_frame = pd.DataFrame(
                {
                    "Symbol": metadata.symbols,
                    "Date": metadata.date,
                    "gnn_risk_score": predictions,
                }
            )

            if metadata.is_classification:
                feature_frame["gnn_risk_class"] = (feature_frame["gnn_risk_score"] >= 0.5).astype(int)
                feature_frame["gnn_risk_margin"] = (
                    feature_frame["gnn_risk_score"] - 0.5
                ).abs() * 2.0
            else:
                feature_frame["gnn_risk_zscore"] = (
                    feature_frame["gnn_risk_score"] - feature_frame["gnn_risk_score"].mean()
                ) / (feature_frame["gnn_risk_score"].std(ddof=0) + 1e-6)

            if mc_stats is not None:
                if mc_stats.get("mean") is not None:
                    feature_frame["gnn_mc_mean"] = mc_stats["mean"]
                if mc_stats.get("std") is not None:
                    feature_frame["gnn_mc_std"] = mc_stats["std"]
                if mc_stats.get("entropy") is not None:
                    feature_frame["gnn_mc_entropy"] = mc_stats["entropy"]

            feature_frame["gnn_graph_density"] = metadata.graph_density
            feature_frame["gnn_target_std"] = metadata.target_std
            feature_frames.append(feature_frame)

        if not feature_frames:
            return pd.DataFrame()

        result = pd.concat(feature_frames, ignore_index=True)
        result.sort_values(["Symbol", "Date"], inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _check_dependencies(self) -> None:
        if torch is None or not isinstance(nn, type):
            raise ImportError("PyTorch is required for the GNN risk predictor")
        if Data is None or GCNConv is None or add_self_loops is None:
            raise ImportError("torch-geometric is required for the GNN risk predictor")
        if GraphRiskNet is None:
            raise ImportError("GraphRiskNet could not be constructed because dependencies are missing")

    def _resolve_target_column(self, df: pd.DataFrame) -> Optional[str]:
        candidates: Tuple[Optional[str], ...] = (self.target_column,) + tuple(self.fallback_target_columns)
        for column in candidates:
            if not column:
                continue
            if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                non_na = df[column].dropna()
                if not non_na.empty:
                    return column
        return None

    def _prepare_graph_data(self, window_df: pd.DataFrame) -> Optional[Tuple[object, GNNMetadata]]:
        if window_df["Symbol"].nunique() < 3:
            return None

        latest_date = window_df["Date"].max()
        latest_slice = window_df[window_df["Date"] == latest_date].copy()
        if latest_slice.empty:
            return None

        latest_slice.sort_values(["Symbol", "Date"], inplace=True)
        latest_slice = latest_slice.groupby("Symbol", as_index=False).last()

        if "Returns" in window_df.columns:
            stats = (
                window_df.groupby("Symbol")["Returns"]
                .agg(mean_return="mean", vol_return="std", min_return="min", max_return="max")
                .reset_index()
            )
            latest_slice = latest_slice.merge(stats, on="Symbol", how="left")

        target_column = self._resolve_target_column(latest_slice)
        if target_column is None:
            self.logger.debug("Unable to locate numeric risk target for date %s", latest_date)
            return None

        corr_matrix = self._correlation_matrix(window_df)
        if corr_matrix is None:
            return None

        symbols = latest_slice["Symbol"].tolist()
        corr_matrix = corr_matrix.reindex(index=symbols, columns=symbols)

        edge_index, edge_weight, density = self._build_edges(corr_matrix)
        if edge_index is None or edge_weight is None:
            return None

        numeric_cols = [
            col
            for col in latest_slice.columns
            if col not in {"Symbol", "Date", target_column} and pd.api.types.is_numeric_dtype(latest_slice[col])
        ]
        if not numeric_cols:
            self.logger.debug("No numeric features available for GNN on %s", latest_date)
            return None

        feature_matrix = latest_slice[numeric_cols].fillna(0.0).to_numpy(dtype=np.float32)
        feature_matrix = self._standardize(feature_matrix)
        x_tensor = torch.tensor(feature_matrix, dtype=torch.float32)

        target_series = latest_slice[target_column].astype(float).fillna(0.0)
        y_tensor = torch.tensor(target_series.to_numpy(dtype=np.float32), dtype=torch.float32)

        unique_targets = np.unique(np.round(target_series.dropna(), decimals=4))
        is_classification = False
        if len(unique_targets) <= 2:
            rounded = {int(val) for val in np.round(unique_targets).astype(int)}
            if rounded.issubset({0, 1}):
                is_classification = True
                y_tensor = y_tensor.clamp(0.0, 1.0)

        data = Data(
            x=x_tensor,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y_tensor,
        )

        metadata = GNNMetadata(
            symbols=symbols,
            date=latest_date,
            is_classification=is_classification,
            target_column=target_column,
            graph_density=density,
            target_std=float(target_series.std(ddof=0)),
        )
        return data, metadata

    def _correlation_matrix(self, window_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if "Returns" not in window_df.columns:
            return None

        returns_wide = window_df.pivot_table(index="Date", columns="Symbol", values="Returns", aggfunc="mean")
        returns_wide = returns_wide.sort_index().fillna(method="ffill").dropna(axis=1, how="all")

        if returns_wide.shape[1] < 3 or returns_wide.shape[0] < max(10, self.window // 3):
            return None

        corr_matrix = returns_wide.corr().fillna(0.0)
        return corr_matrix

    def _build_edges(
        self,
        corr_matrix: pd.DataFrame,
    ) -> Tuple[Optional[object], Optional[object], float]:
        symbols = corr_matrix.index.tolist()
        num_nodes = len(symbols)
        if num_nodes == 0:
            return None, None, 0.0

        edge_pairs: List[Tuple[int, int]] = []
        edge_weights: List[float] = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                weight = float(corr_matrix.iat[i, j])
                if np.isnan(weight):
                    continue
                if abs(weight) >= self.correlation_threshold:
                    edge_pairs.append((i, j))
                    edge_pairs.append((j, i))
                    edge_weights.append(weight)
                    edge_weights.append(weight)

        density = 0.0
        if num_nodes > 1:
            density = len(edge_pairs) / float(num_nodes * (num_nodes - 1))

        if torch is None:
            return None, None, density

        if edge_pairs:
            edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.tensor([], dtype=torch.float32)

        if add_self_loops is not None:
            edge_index, edge_weight = add_self_loops(
                edge_index=edge_index,
                edge_weight=edge_weight,
                fill_value=1.0,
                num_nodes=num_nodes,
            )

        return edge_index, edge_weight, density

    def _standardize(self, matrix: np.ndarray) -> np.ndarray:
        mean = matrix.mean(axis=0, keepdims=True)
        std = matrix.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        standardized = (matrix - mean) / std
        return np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_model(self, input_dim: int) -> GraphRiskNet:
        assert GraphRiskNet is not None  # For type checkers
        model = GraphRiskNet(
            input_dim=input_dim,
            hidden_channels=self.hidden_channels,
            gnn_type=self.gnn_type,
            heads=self.heads,
            dropout=self.dropout,
        )
        return model

    def _train_and_predict(
        self, data: object, metadata: GNNMetadata
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        assert torch is not None  # guarded by dependency checks

        torch.manual_seed(self.random_state)
        model = self._build_model(data.num_node_features).to(self.device)
        data = data.to(self.device)

        if metadata.is_classification:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        best_loss = float("inf")
        best_state: Optional[Dict[str, object]] = None
        patience_counter = 0

        for _ in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, getattr(data, "edge_weight", None))
            target = data.y
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                validation_output = model(
                    data.x, data.edge_index, getattr(data, "edge_weight", None)
                )
                validation_loss = criterion(validation_output, target).item()

            if validation_loss + self.tolerance < best_loss:
                best_loss = validation_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index, getattr(data, "edge_weight", None))
            if metadata.is_classification:
                predictions = torch.sigmoid(logits).cpu().numpy()
            else:
                predictions = logits.cpu().numpy()

        mc_stats: Optional[Dict[str, np.ndarray]] = None
        if (
            monte_carlo_dropout is not None
            and self.mc_dropout_samples > 0
            and self.dropout > 0
        ):

            def _forward():
                output = model(
                    data.x,
                    data.edge_index,
                    getattr(data, "edge_weight", None),
                )
                if metadata.is_classification:
                    output = torch.sigmoid(output)
                return output

            try:
                mean, std = monte_carlo_dropout(
                    model,
                    _forward,
                    samples=self.mc_dropout_samples,
                )
                entropy = None
                if metadata.is_classification:
                    entropy = _binary_entropy(mean)
                mc_stats = {
                    "mean": mean,
                    "std": std,
                    "entropy": entropy,
                }
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning(
                    "Monte Carlo dropout failed for graph model on %s: %s",
                    metadata.date,
                    exc,
                )

        return predictions, mc_stats


def _binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    probs = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return -(probs * np.log(probs) + (1 - probs) * np.log(1 - probs))


__all__ = ["GNNRiskPredictor"]
