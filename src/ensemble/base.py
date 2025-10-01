"""Base abstractions for ensemble strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


class BaseEnsemble(ABC):
    """Abstract base class for combining model probability outputs."""

    def __init__(self, model_names: Optional[Sequence[str]] = None) -> None:
        self.model_names = list(model_names) if model_names else []
        self.weights: MutableMapping[str, float] = {}

    @abstractmethod
    def fit(self, model_probabilities: Mapping[str, Iterable[float]], **kwargs: object) -> "BaseEnsemble":
        """Learn ensemble parameters from model probabilities."""

    @abstractmethod
    def predict(self, model_probabilities: Mapping[str, Iterable[float]], **kwargs: object) -> np.ndarray:
        """Combine probabilities into a single risk estimate."""

    def get_weights(self) -> Mapping[str, float]:
        return dict(self.weights)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _ordered_names(self, mapping: Mapping[str, Iterable[float]]) -> Tuple[str, ...]:
        if self.model_names:
            names = tuple(name for name in self.model_names if name in mapping)
        else:
            names = tuple(mapping.keys())
            self.model_names = list(names)
        if not names:
            raise ValueError("No model probabilities provided to the ensemble")
        return names

    def _stack_probabilities(
        self,
        mapping: Mapping[str, Iterable[float]],
    ) -> Tuple[Tuple[str, ...], np.ndarray]:
        names = self._ordered_names(mapping)
        arrays = []
        expected_length = None
        for name in names:
            values = np.asarray(list(mapping[name]), dtype=float)
            if expected_length is None:
                expected_length = len(values)
            elif len(values) != expected_length:
                raise ValueError("All probability arrays must share the same length")
            arrays.append(values)
        stacked = np.vstack(arrays)
        return names, stacked

    def _normalize_weight_vector(self, weights: np.ndarray) -> np.ndarray:
        weights = np.clip(weights, a_min=0.0, a_max=None)
        total = weights.sum()
        if total <= 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / total
        return weights

    def _vector_from_mapping(
        self,
        names: Sequence[str],
        weight_mapping: Mapping[str, float],
        floor: float = 0.0,
    ) -> np.ndarray:
        return np.array([
            max(float(weight_mapping.get(name, 0.0)), floor) for name in names
        ])
