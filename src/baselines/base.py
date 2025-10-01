"""Abstract baseline interfaces and result containers for MEWS."""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Container for storing baseline outputs.

    Attributes:
        name: Human readable identifier for the baseline.
        predictions: DataFrame containing risk probabilities or thresholds.
        metadata: Optional dictionary with diagnostic information.
    """

    name: str
    predictions: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the result into a dictionary friendly format."""

        return {
            "name": self.name,
            "predictions": self.predictions.to_dict(orient="records"),
            "metadata": self.metadata,
        }


class BaseBaseline(abc.ABC):
    """Base class for risk baselines.

    Concrete implementations should override :meth:`run` to produce a
    :class:`BaselineResult`. Helper utilities for column validation are provided to
    encourage consistent error handling across baselines.
    """

    name: str = "baseline"

    def __init__(self) -> None:
        self.logger = LOGGER

    @abc.abstractmethod
    def run(self, df: pd.DataFrame, **kwargs: Any) -> BaselineResult:
        """Execute the baseline on the provided dataframe."""

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _require_columns(self, df: pd.DataFrame, columns: Iterable[str]) -> None:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Dataframe is missing required columns: {missing}")

    def _prepare_dataframe(
        self,
        df: pd.DataFrame,
        date_col: str,
        symbol_col: str,
    ) -> pd.DataFrame:
        """Return dataframe sorted by symbol/date with datetime index."""

        working = df.copy()
        working[date_col] = pd.to_datetime(working[date_col])
        working.sort_values([symbol_col, date_col], inplace=True)
        return working
