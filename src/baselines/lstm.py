"""LSTM-based baseline producing risk probabilities from temporal features."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import BaseBaseline, BaselineResult

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from src.uncertainty.monte_carlo_dropout import monte_carlo_dropout
except ImportError:  # pragma: no cover
    monte_carlo_dropout = None  # type: ignore


class _LSTMClassifier(nn.Module):  # pragma: no cover - small network container
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        logits = self.linear(last_hidden)
        return logits


class LSTMBaseline(BaseBaseline):
    """Sequence model baseline returning risk probabilities."""

    name = "lstm_risk_predictor"

    def __init__(
        self,
        target_col: str = "Risk_Label",
        feature_cols: Optional[Sequence[str]] = None,
        returns_col: str = "Returns",
        sequence_length: int = 30,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        mc_dropout_samples: int = 0,
    ) -> None:
        super().__init__()
        self.target_col = target_col
        self.feature_cols = list(feature_cols) if feature_cols else [returns_col]
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mc_dropout_samples = mc_dropout_samples

    def run(
        self,
        df: pd.DataFrame,
        symbol_col: str = "Symbol",
        date_col: str = "Date",
        **_: Any,
    ) -> BaselineResult:
        if torch is None or nn is None:
            raise ImportError("PyTorch is required for the LSTM baseline")

        required_columns = set(
            self.feature_cols + [self.target_col, symbol_col, date_col]
        )
        self._require_columns(df, required_columns)
        working = self._prepare_dataframe(df, date_col=date_col, symbol_col=symbol_col)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

        outputs: List[pd.DataFrame] = []
        metadata: Dict[str, Dict[str, Any]] = {}

        for symbol, group in working.groupby(symbol_col):
            sequences, targets, timestamps = self._build_sequences(group, date_col)
            if not sequences:
                continue

            X_tensor = torch.tensor(
                np.stack(sequences), dtype=torch.float32, device=device
            )
            y_tensor = torch.tensor(
                np.array(targets), dtype=torch.float32, device=device
            ).unsqueeze(1)

            model = _LSTMClassifier(
                input_size=X_tensor.shape[-1],
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
            ).to(device)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

            history: List[float] = []
            sample_count = X_tensor.shape[0]

            for epoch in range(self.epochs):
                permutation = torch.randperm(sample_count, device=device)
                epoch_loss = 0.0

                for start in range(0, sample_count, self.batch_size):
                    idx = permutation[start : start + self.batch_size]
                    batch_x = X_tensor.index_select(0, idx)
                    batch_y = y_tensor.index_select(0, idx)

                    optimizer.zero_grad()
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += float(loss.item()) * batch_x.size(0)

                epoch_loss /= sample_count
                history.append(epoch_loss)

            model.eval()
            with torch.no_grad():
                logits = model(X_tensor)
                probabilities_tensor = torch.sigmoid(logits).squeeze(1)
                probabilities = probabilities_tensor.cpu().numpy()

            mc_mean = None
            mc_std = None
            mc_entropy = None
            if (
                monte_carlo_dropout is not None
                and self.mc_dropout_samples > 0
                and self.dropout > 0
            ):

                def _forward():
                    return torch.sigmoid(model(X_tensor)).squeeze(1)

                try:
                    mc_mean, mc_std = monte_carlo_dropout(
                        model,
                        _forward,
                        samples=self.mc_dropout_samples,
                    )
                    mc_entropy = _binary_entropy(mc_mean)
                except Exception as exc:  # pragma: no cover - defensive
                    self.logger.warning(
                        "Monte Carlo dropout failed for %s: %s", symbol, exc
                    )
                    mc_mean = mc_std = mc_entropy = None

            result_df = pd.DataFrame(
                {
                    symbol_col: symbol,
                    date_col: timestamps,
                    "lstm_risk_probability": probabilities,
                }
            )
            if mc_mean is not None and mc_std is not None:
                result_df["lstm_mc_mean"] = mc_mean
                result_df["lstm_mc_std"] = mc_std
                if mc_entropy is not None:
                    result_df["lstm_mc_entropy"] = mc_entropy
            outputs.append(result_df)
            metadata[symbol] = {
                "sequences": int(sample_count),
                "final_loss": history[-1] if history else math.nan,
                "epochs": self.epochs,
                "features": list(self.feature_cols),
            }
            if mc_mean is not None and mc_std is not None:
                metadata[symbol]["mc_dropout_samples"] = self.mc_dropout_samples
                metadata[symbol]["mc_mean"] = float(np.mean(mc_mean))
                metadata[symbol]["mc_std"] = float(np.mean(mc_std))

        if not outputs:
            raise ValueError(
                "LSTM baseline could not generate predictions for any symbols"
            )

        predictions = pd.concat(outputs, ignore_index=True)
        metadata["sequence_length"] = self.sequence_length

        return BaselineResult(
            name=self.name, predictions=predictions, metadata=metadata
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_sequences(
        self,
        group: pd.DataFrame,
        date_col: str,
    ) -> Tuple[List[np.ndarray], List[float], List[pd.Timestamp]]:
        group_sorted = group.sort_values(date_col).reset_index(drop=True)
        features = group_sorted[self.feature_cols].copy()
        features = features.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(features.values)

        targets = group_sorted[self.target_col].astype(float)
        timestamps = pd.to_datetime(group_sorted[date_col])

        sequences: List[np.ndarray] = []
        labels: List[float] = []
        label_timestamps: List[pd.Timestamp] = []

        for idx in range(self.sequence_length, len(group_sorted)):
            target = targets.iloc[idx]
            if np.isnan(target):
                continue
            window = scaled[idx - self.sequence_length : idx]
            sequences.append(window)
            labels.append(float(target))
            label_timestamps.append(timestamps.iloc[idx])

        return sequences, labels, label_timestamps


def _binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    probs = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return -(probs * np.log(probs) + (1 - probs) * np.log(1 - probs))
