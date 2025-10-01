"""Monte Carlo dropout helpers for PyTorch models."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore

if torch is not None:

    def enable_dropout_layers(model: Any) -> None:
        """Activate dropout layers during evaluation.

        This utility forces all dropout modules into training mode, enabling
        stochastic forward passes without altering the global model state
        outside the returned context.
        """

        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    @contextmanager
    def _dropout_active(model: Any):
        was_training = model.training
        model.train()
        enable_dropout_layers(model)
        try:
            yield
        finally:
            if was_training:
                model.train()
            else:
                model.eval()

    def monte_carlo_dropout(
        model: Any,
        forward_fn: Callable[[], Any],
        *,
        samples: int = 30,
        disable_grad: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute Monte Carlo dropout sampling.

        Args:
            model: Neural network containing dropout layers.
            forward_fn: Callable executing a single forward pass and returning
                probabilities/logits as a 1D tensor.
            samples: Number of stochastic samples to draw.
            disable_grad: If ``True`` no gradients are tracked.

        Returns:
            Tuple of (mean, standard deviation) over the sampled predictions.
        """

        if samples <= 0:
            raise ValueError("samples must be positive for Monte Carlo dropout")

        if disable_grad:
            gradient_context = torch.no_grad
        else:

            @contextmanager
            def identity_context():
                yield

            gradient_context = identity_context

        with _dropout_active(model):
            with gradient_context():
                draws = []
                for _ in range(samples):
                    output = forward_fn()
                    if not isinstance(output, torch.Tensor):
                        raise TypeError("forward_fn must return a torch.Tensor")
                    draws.append(output.detach().cpu().numpy())

        stacked = np.stack(draws, axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        return mean, std


else:  # pragma: no cover - torch unavailable

    def enable_dropout_layers(model):  # type: ignore[override]
        raise ImportError("PyTorch is required for Monte Carlo dropout utilities")

    def monte_carlo_dropout(*args, **kwargs):  # type: ignore[override]
        raise ImportError("PyTorch is required for Monte Carlo dropout utilities")
