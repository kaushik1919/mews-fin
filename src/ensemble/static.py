"""Static weighted ensemble for MEWS model probabilities."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import numpy as np

from .base import BaseEnsemble


class StaticWeightedEnsemble(BaseEnsemble):
    """Combine model probabilities using static weights."""

    def __init__(
        self,
        weights: Optional[Mapping[str, float]] = None,
        floor_weight: float = 1e-3,
    ) -> None:
        super().__init__(model_names=list(weights.keys()) if weights else None)
        self._provided_weights = dict(weights) if weights else None
        self.floor_weight = floor_weight

    def fit(
        self,
        model_probabilities: Mapping[str, Iterable[float]],
        model_scores: Optional[Mapping[str, float]] = None,
        **_: object,
    ) -> "StaticWeightedEnsemble":
        names, _ = self._stack_probabilities(model_probabilities)

        if self._provided_weights:
            weight_vector = self._vector_from_mapping(
                names, self._provided_weights, self.floor_weight
            )
        elif model_scores:
            weight_vector = self._vector_from_mapping(
                names, model_scores, self.floor_weight
            )
        else:
            weight_vector = np.ones(len(names), dtype=float)

        weight_vector = self._normalize_weight_vector(weight_vector)
        self.weights = {
            name: float(weight) for name, weight in zip(names, weight_vector)
        }
        return self

    def predict(
        self,
        model_probabilities: Mapping[str, Iterable[float]],
        **_: object,
    ) -> np.ndarray:
        names, stacked = self._stack_probabilities(model_probabilities)
        if not self.weights:
            # Default to equal weighting if fit was not called.
            weight_vector = np.ones(len(names), dtype=float) / len(names)
        else:
            weight_vector = self._vector_from_mapping(names, self.weights)
            weight_vector = self._normalize_weight_vector(weight_vector)

        combined = np.average(stacked, axis=0, weights=weight_vector)
        return combined
