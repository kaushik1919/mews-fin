"""Case study runner that replays historical crises using MEWS predictions."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .scenarios import CaseStudyScenario

LOGGER = logging.getLogger(__name__)


@dataclass
class CaseStudyResult:
    """Outputs generated for a case study scenario."""

    scenario: CaseStudyScenario
    data_window: Tuple[pd.Timestamp, pd.Timestamp]
    symbols: Sequence[str]
    total_observations: int
    warning_events: int
    downturn_events: int
    combined_events: int
    top_features: List[Tuple[str, int]]
    plot_path: Optional[Path]
    report_path: Optional[Path]


class CaseStudyRunner:
    """Runs predefined crisis case studies and exports research artifacts."""

    def __init__(
        self,
        data_path: Optional[Path | str] = None,
        predictions_path: Optional[Path | str] = None,
        feature_groups_path: Optional[Path | str] = None,
        output_dir: Optional[Path | str] = None,
    ) -> None:
        self.data_path = Path(data_path or "data/integrated_dataset.csv")
        self.predictions_path = Path(predictions_path or "outputs/risk_predictions.csv")
        self.feature_groups_path = Path(
            feature_groups_path or "data/feature_groups.json"
        )
        self.output_dir = Path(output_dir or "outputs/case_studies")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.feature_groups = self._load_feature_groups()
        self.numeric_feature_columns: List[str] = []
        self.matplotlib_available = self._check_matplotlib()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_case_study(self, scenario: CaseStudyScenario) -> CaseStudyResult:
        """Execute a single case study and export plots and reports."""

        raw_data = self._load_dataframe(self.data_path)
        predictions = self._load_dataframe(self.predictions_path)

        if raw_data.empty:
            raise FileNotFoundError(
                f"Integrated dataset not found or empty at {self.data_path.as_posix()}"
            )

        if predictions.empty:
            LOGGER.warning(
                "Risk predictions CSV missing or empty (%s); proceeding with data-only "
                "analysis.",
                self.predictions_path,
            )

        merged = self._prepare_dataset(raw_data, predictions)
        scenario_df, window, symbols, window_adjusted = self._filter_scenario_window(
            merged, scenario
        )

        if scenario_df.empty:
            raise ValueError(
                f"No data available for scenario '{scenario.name}' between {scenario.start_date} "
                f"and {scenario.end_date} with symbols {symbols}."
            )

        events_df = self._tag_events(scenario_df, scenario)
        triggers, feature_counts = self._identify_feature_triggers(
            scenario_df, events_df, scenario
        )

        plot_path = self._generate_plot(scenario_df, events_df, triggers, scenario)
        report_path = self._generate_report(
            scenario,
            scenario_df,
            events_df,
            feature_counts,
            window,
            symbols,
            triggers,
            plot_path,
            window_adjusted,
        )

        warning_events = int(events_df["warning_flag"].sum())
        downturn_events = int(events_df["downturn_flag"].sum())
        combined_events = int(events_df["combined_flag"].sum())

        result = CaseStudyResult(
            scenario=scenario,
            data_window=window,
            symbols=symbols,
            total_observations=len(scenario_df),
            warning_events=warning_events,
            downturn_events=downturn_events,
            combined_events=combined_events,
            top_features=feature_counts.most_common(),
            plot_path=plot_path,
            report_path=report_path,
        )
        return result

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _load_feature_groups(self) -> Dict[str, List[str]]:
        if not self.feature_groups_path.exists():
            LOGGER.warning(
                "Feature groups JSON not found at %s; feature annotations will be limited.",
                self.feature_groups_path,
            )
            return {}

        with self.feature_groups_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        feature_groups = {
            group: [col for col in cols if isinstance(col, str)]
            for group, cols in payload.items()
        }
        all_features = sorted({col for cols in feature_groups.values() for col in cols})
        self.numeric_feature_columns = all_features
        return feature_groups

    def _check_matplotlib(self) -> bool:
        try:
            import matplotlib.pyplot as plt  # noqa: F401
            import seaborn as sns  # noqa: F401

            return True
        except ImportError:
            LOGGER.warning(
                "Matplotlib/Seaborn not available; case study plots will be skipped."
            )
            return False

    def _load_dataframe(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            LOGGER.warning("Data file %s does not exist", path)
            return pd.DataFrame()

        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - IO/parsing errors
            LOGGER.error("Failed to read %s: %s", path, exc)
            return pd.DataFrame()

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"])
        return df

    def _prepare_dataset(
        self, data_df: pd.DataFrame, predictions_df: pd.DataFrame
    ) -> pd.DataFrame:
        df = data_df.copy()

        if "Symbol" not in df.columns:
            raise KeyError("Integrated dataset must contain a 'Symbol' column")

        df = df.sort_values(["Symbol", "Date"])

        if not predictions_df.empty:
            pred_df = predictions_df.copy()
            pred_df = pred_df.rename(columns={"Risk_Label": "Risk_Label_Pred"})
            merge_cols = [
                col for col in pred_df.columns if col not in {"Date", "Symbol"}
            ]
            merged = pd.merge(
                df,
                pred_df,
                on=["Date", "Symbol"],
                how="left",
                suffixes=("", "_pred"),
            )

            for col in merge_cols:
                candidate = f"{col}_pred"
                if candidate in merged.columns and col not in df.columns:
                    merged[col] = merged[candidate]
                if candidate in merged.columns:
                    merged = merged.drop(columns=[candidate])
            df = merged

        df = self._augment_metrics(df)
        return df

    def _augment_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Daily return proxy
        if "Returns" in df.columns:
            df["Actual_Return"] = df["Returns"].fillna(0.0)
        else:
            df["Actual_Return"] = (
                df.groupby("Symbol")["Close"].pct_change().fillna(0.0)
                if "Close" in df.columns
                else 0.0
            )

        if "Close" in df.columns:
            df["Rolling_Max_Close"] = df.groupby("Symbol")["Close"].cummax()
            df["Drawdown"] = df["Close"] / df["Rolling_Max_Close"] - 1.0
        else:
            df["Drawdown"] = (
                df.groupby("Symbol")["Actual_Return"].cumsum()
                if "Actual_Return" in df.columns
                else 0.0
            )
        df["Drawdown"] = df["Drawdown"].fillna(0.0)

        if "Risk_Probability" not in df.columns and "Risk_Score" in df.columns:
            df["Risk_Probability"] = df["Risk_Score"].clip(0, 1)

        if "Risk_Prediction" not in df.columns and "Risk_Label_Pred" in df.columns:
            df["Risk_Prediction"] = df["Risk_Label_Pred"].fillna(0).astype(int)

        df["Actual_Return"] = df["Actual_Return"].astype(float)
        df["Drawdown"] = df["Drawdown"].astype(float)
        df = df.drop(columns=["Rolling_Max_Close"], errors="ignore")
        return df

    def _filter_scenario_window(
        self, df: pd.DataFrame, scenario: CaseStudyScenario
    ) -> Tuple[pd.DataFrame, Tuple[pd.Timestamp, pd.Timestamp], Sequence[str], bool]:
        min_date = df["Date"].min()
        max_date = df["Date"].max()

        scenario_start = pd.Timestamp(scenario.start_date)
        scenario_end = pd.Timestamp(scenario.end_date)

        start = max(min_date, scenario_start)
        end = min(max_date, scenario_end)
        adjusted = False

        if start > end:
            LOGGER.warning(
                "Scenario '%s' window %s–%s does not overlap with dataset (%s–%s); using available data range instead.",
                scenario.name,
                scenario_start.date(),
                scenario_end.date(),
                min_date.date(),
                max_date.date(),
            )
            start, end = min_date, max_date
            adjusted = True

        subset = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()

        if scenario.symbols:
            symbols = [
                sym for sym in scenario.symbols if sym in subset["Symbol"].unique()
            ]
            if not symbols:
                symbols = list(subset["Symbol"].unique())
            subset = subset[subset["Symbol"].isin(symbols)]
        else:
            symbols = list(subset["Symbol"].unique())

        subset = subset.sort_values(["Symbol", "Date"]).reset_index(drop=True)
        return subset, (start, end), symbols, adjusted

    # ------------------------------------------------------------------
    # Event detection and feature triggers
    # ------------------------------------------------------------------
    def _tag_events(
        self, df: pd.DataFrame, scenario: CaseStudyScenario
    ) -> pd.DataFrame:
        events = df.copy()

        prob_col = "Risk_Probability" if "Risk_Probability" in events.columns else None
        pred_col = "Risk_Prediction" if "Risk_Prediction" in events.columns else None

        if prob_col is None and pred_col is None:
            LOGGER.warning(
                "Scenario %s: risk probabilities not found; using Actual_Return for warnings.",
                scenario.name,
            )

        warning_mask = np.zeros(len(events), dtype=bool)
        if prob_col is not None:
            warning_mask = warning_mask | (events[prob_col] >= scenario.risk_threshold)
        if pred_col is not None:
            warning_mask = warning_mask | (events[pred_col] == 1)

        risk_label_series = (
            events["Risk_Label"] if "Risk_Label" in events.columns else None
        )
        downturn_mask = (
            (events["Drawdown"] <= -0.05)
            | (events["Actual_Return"] <= -0.02)
            | ((risk_label_series == 1) if risk_label_series is not None else False)
        )

        events["warning_flag"] = warning_mask
        events["downturn_flag"] = downturn_mask
        events["combined_flag"] = warning_mask & downturn_mask
        return events

    def _identify_feature_triggers(
        self,
        scenario_df: pd.DataFrame,
        events_df: pd.DataFrame,
        scenario: CaseStudyScenario,
    ) -> Tuple[List[Dict[str, object]], Counter]:
        if scenario_df.empty:
            return [], Counter()

        candidate_cols = [
            col
            for col in self.numeric_feature_columns
            if col in scenario_df.columns
            and pd.api.types.is_numeric_dtype(scenario_df[col])
        ]

        if not candidate_cols:
            numeric_cols = scenario_df.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            candidate_cols = [
                col
                for col in numeric_cols
                if col not in {"Risk_Probability", "Risk_Prediction"}
            ]

        stats_mean = scenario_df[candidate_cols].mean()
        stats_std = scenario_df[candidate_cols].std().replace(0, np.nan)

        warning_events = events_df[events_df["warning_flag"]].copy()

        triggers: List[Dict[str, object]] = []
        feature_counter: Counter = Counter()

        for _, row in warning_events.iterrows():
            zscores: List[Tuple[str, float]] = []
            for col in candidate_cols:
                value = row.get(col)
                if pd.isna(value):
                    continue
                mean = stats_mean[col]
                std = stats_std[col]
                if pd.isna(std) or std == 0:
                    continue
                z = abs((value - mean) / std)
                if np.isfinite(z):
                    zscores.append((col, float(z)))

            zscores.sort(key=lambda item: item[1], reverse=True)
            top_features = [name for name, _ in zscores[: scenario.top_feature_count]]
            feature_counter.update(top_features)

            triggers.append(
                {
                    "Date": row["Date"],
                    "Symbol": row["Symbol"],
                    "features": top_features,
                    "zscores": zscores[: scenario.top_feature_count],
                }
            )

        return triggers, feature_counter

    # ------------------------------------------------------------------
    # Visualization and reporting
    # ------------------------------------------------------------------
    def _generate_plot(
        self,
        scenario_df: pd.DataFrame,
        events_df: pd.DataFrame,
        triggers: List[Dict[str, object]],
        scenario: CaseStudyScenario,
    ) -> Optional[Path]:
        if not self.matplotlib_available or scenario_df.empty:
            return None

        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style("whitegrid")

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        palette = sns.color_palette(
            "Set2", n_colors=len(scenario_df["Symbol"].unique())
        )
        symbol_colors = {
            symbol: palette[idx % len(palette)]
            for idx, symbol in enumerate(sorted(scenario_df["Symbol"].unique()))
        }

        for symbol, symbol_df in scenario_df.groupby("Symbol"):
            color = symbol_colors[symbol]
            axes[0].plot(
                symbol_df["Date"],
                symbol_df.get("Risk_Probability", 0.0),
                label=f"{symbol} Risk Probability",
                color=color,
                linewidth=1.8,
            )
            axes[1].plot(
                symbol_df["Date"],
                symbol_df["Drawdown"] * 100,
                label=f"{symbol} Drawdown",
                color=color,
                linewidth=1.2,
            )

        warning_points = events_df[events_df["warning_flag"]]
        axes[0].scatter(
            warning_points["Date"],
            warning_points.get("Risk_Probability", 0.0),
            color="red",
            marker="^",
            s=40,
            label="Risk Warning",
            alpha=0.7,
        )

        combined_points = events_df[events_df["combined_flag"]]
        axes[1].scatter(
            combined_points["Date"],
            combined_points["Drawdown"] * 100,
            color="black",
            marker="o",
            s=35,
            label="Warning & Downturn",
            alpha=0.6,
        )

        for ax in axes:
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

        axes[0].set_ylabel("Risk Probability")
        axes[0].set_title(f"{scenario.name} — Predicted Risk vs Actual Downturn")
        axes[0].legend(loc="upper left")

        axes[1].set_ylabel("Drawdown (%)")
        axes[1].legend(loc="lower left")

        for marker_date, label in scenario.marker_dates():
            marker_ts = pd.Timestamp(marker_date)
            for ax in axes:
                ax.axvline(
                    marker_ts, color="gray", linestyle=":", linewidth=1.0, alpha=0.8
                )
                ax.text(
                    marker_ts,
                    ax.get_ylim()[1],
                    label,
                    rotation=90,
                    verticalalignment="bottom",
                    horizontalalignment="right",
                    fontsize=9,
                    color="dimgray",
                )

        if triggers:
            annotated = triggers[: min(len(triggers), 5)]
            for entry in annotated:
                date = pd.Timestamp(entry["Date"])
                symbol = entry["Symbol"]
                features = entry["features"][:2]
                axes[0].annotate(
                    ", ".join(features),
                    xy=(date, 0.95),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color="darkred",
                    arrowprops=dict(arrowstyle="-", color="darkred", lw=0.8),
                )
                axes[1].annotate(
                    symbol,
                    xy=(date, 0),
                    xytext=(0, -15),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color="black",
                )

        axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
        axes[1].xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(axes[1].xaxis.get_major_locator())
        )
        fig.tight_layout()

        output_path = self.output_dir / f"{scenario.slug}_risk_vs_downturn.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def _generate_report(
        self,
        scenario: CaseStudyScenario,
        scenario_df: pd.DataFrame,
        events_df: pd.DataFrame,
        feature_counts: Counter,
        window: Tuple[pd.Timestamp, pd.Timestamp],
        symbols: Sequence[str],
        triggers: List[Dict[str, object]],
        plot_path: Optional[Path],
        window_adjusted: bool,
    ) -> Path:
        report_path = self.output_dir / f"{scenario.slug}_report.md"

        start, end = window
        warning_events = int(events_df["warning_flag"].sum())
        downturn_events = int(events_df["downturn_flag"].sum())
        combined_events = int(events_df["combined_flag"].sum())

        top_features_md = (
            "\n".join(
                f"- **{feature}**: {count} warnings"
                for feature, count in feature_counts.most_common(10)
            )
            or "- _No feature triggers recorded._"
        )

        trigger_rows = []
        for entry in triggers[:10]:
            date = pd.Timestamp(entry["Date"]).strftime("%Y-%m-%d")
            symbol = entry["Symbol"]
            features = (
                ", ".join(entry["features"][: scenario.top_feature_count]) or "N/A"
            )
            trigger_rows.append(f"| {date} | {symbol} | {features} |")

        trigger_table = (
            (
                "| Date | Symbol | Top Features |\n| --- | --- | --- |\n"
                + "\n".join(trigger_rows)
            )
            if trigger_rows
            else "_No high-risk warnings detected within the scenario window._"
        )

        plot_section = (
            f"![Risk vs Downturn Plot]({plot_path.as_posix()})"
            if plot_path
            else "_Plot unavailable._"
        )

        milestone_lines = (
            "\n".join(
                f"- {marker} — {label}" for marker, label in scenario.marker_dates()
            )
            if scenario.crisis_markers
            else "- _(No milestone annotations provided.)_"
        )

        data_note = (
            "_Original crisis window partially unavailable; analysis uses closest available dataset range._"
            if window_adjusted
            else ""
        )

        report_contents = f"""# {scenario.name} Case Study

**Study window:** {start.date()} → {end.date()}  \
**Symbols analyzed:** {', '.join(symbols)}  \
**Observations:** {len(scenario_df)} rows
{data_note}

## Crisis Narrative
{scenario.description}

Key policy and market milestones:
{milestone_lines}

## Warning Statistics
- Risk warnings (model-based): **{warning_events}**
- Observed downturn signals: **{downturn_events}**
- Warnings aligned with downturns: **{combined_events}**

## Feature Triggers
{top_features_md}

### Warning Events Breakdown
{trigger_table}

## Predicted Risk vs Actual Downturn
{plot_section}

## Methodology Snapshot
- Risk probabilities sourced from `{self.predictions_path.as_posix()}`.
- Market features sourced from `{self.data_path.as_posix()}`.
- Drawdown computed as percentage from rolling peak close price.
- Feature triggers ranked by absolute z-score within the scenario window.

_Report generated automatically by `CaseStudyRunner`._
"""

        report_path.write_text(report_contents, encoding="utf-8")
        return report_path
