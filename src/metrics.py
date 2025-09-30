"""Metric utilities for the Market Risk Early Warning System."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class CEWSResult:
    """Structured output for Crisis Early Warning Score computations."""

    score: float
    early_detection_reward: float
    false_alarm_penalty: float
    timeline: pd.DataFrame
    metadata: Dict[str, float]


def compute_cews_score(
    df: pd.DataFrame,
    probability_col: str = "Risk_Probability",
    label_col: str = "Risk_Label",
    date_col: str = "Date",
    symbol_col: Optional[str] = "Symbol",
    threshold: float = 0.5,
    lookback_window: int = 5,
) -> CEWSResult:
    """Compute Crisis Early Warning Score (CEWS) with interpretable components.

    The CEWS metric rewards timely identification of risk events and penalizes
    persistent false alarms. A prediction is considered an "alert" when the
    forecast probability exceeds ``threshold``.

    Args:
        df: DataFrame containing predictions and ground-truth labels.
        probability_col: Column name with model risk probabilities.
        label_col: Column name with binary risk labels (1 = crisis, 0 = stable).
        date_col: Column containing chronological ordering information.
        symbol_col: Optional column defining independent time series (e.g. tickers).
        threshold: Probability threshold that constitutes an alert.
        lookback_window: Number of past days to consider when scoring early
            detections and the forward window used to penalize false alarms.

    Returns:
        ``CEWSResult`` with scalar score, components, and a per-date timeline
        suitable for visualization.
    """

    required_cols = {probability_col, label_col, date_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for CEWS computation: {missing}")

    working_df = df.copy()
    as_str = working_df[date_col].astype(str)
    working_df[date_col] = pd.to_datetime(as_str, errors="coerce", utc=True)
    working_df[date_col] = working_df[date_col].dt.tz_localize(None)
    working_df = working_df.dropna(subset=[date_col])
    working_df = working_df.sort_values(date_col)

    if symbol_col and symbol_col in working_df.columns:
        group_keys = [symbol_col]
    else:
        group_keys = []

    working_df["alert"] = working_df[probability_col] >= threshold
    working_df["event"] = working_df[label_col] == 1

    if group_keys:
        grouped = working_df.groupby(group_keys)
        working_df["alert_rolling"] = grouped["alert"].transform(
            lambda x: x.rolling(window=lookback_window, min_periods=1).max()
        )
        working_df["future_event"] = (
            grouped[label_col]
            .transform(
                lambda x: x[::-1]
                .rolling(window=lookback_window, min_periods=1)
                .max()[::-1]
            )
            .astype(int)
        )
    else:
        working_df["alert_rolling"] = (
            working_df["alert"].rolling(window=lookback_window, min_periods=1).max()
        )
        working_df["future_event"] = (
            working_df[label_col][::-1]
            .rolling(window=lookback_window, min_periods=1)
            .max()[::-1]
            .astype(int)
        )

    working_df["detected_event"] = working_df["event"] & (
        working_df["alert_rolling"] == 1
    )
    working_df["false_alarm"] = working_df["alert"] & (
        working_df["future_event"] == 0
    )

    total_events = float(working_df["event"].sum())
    detected_events = float(working_df["detected_event"].sum())
    positive_alerts = float(working_df["alert"].sum())
    false_alarms = float(working_df["false_alarm"].sum())

    early_detection_reward = (
        detected_events / total_events if total_events > 0 else 0.0
    )
    false_alarm_penalty = (
        false_alarms / positive_alerts if positive_alerts > 0 else 0.0
    )

    raw_score = early_detection_reward - false_alarm_penalty
    score = float(np.clip(raw_score, 0.0, 1.0))

    timeline = (
        working_df.groupby(date_col)
        .agg(
            early_reward=("detected_event", "sum"),
            events=("event", "sum"),
            alerts=("alert", "sum"),
            false_alarms=("false_alarm", "sum"),
        )
        .reset_index()
    )
    timeline["early_reward"] = np.where(
        timeline["events"] > 0,
        timeline["early_reward"] / timeline["events"],
        0.0,
    )
    timeline["false_alarm_penalty"] = np.where(
        timeline["alerts"] > 0,
        timeline["false_alarms"] / timeline["alerts"],
        0.0,
    )
    timeline["cews"] = np.clip(
        timeline["early_reward"] - timeline["false_alarm_penalty"], 0.0, 1.0
    )

    metadata = {
        "total_events": total_events,
        "detected_events": detected_events,
        "false_alarms": false_alarms,
        "positive_alerts": positive_alerts,
        "threshold": threshold,
        "lookback_window": float(lookback_window),
    }

    return CEWSResult(
        score=score,
        early_detection_reward=float(early_detection_reward),
        false_alarm_penalty=float(false_alarm_penalty),
        timeline=timeline,
        metadata=metadata,
    )


__all__ = ["CEWSResult", "compute_cews_score"]
