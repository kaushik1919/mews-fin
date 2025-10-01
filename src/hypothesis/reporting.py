"""Reporting utilities for hypothesis testing outputs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

import pandas as pd

from .results import (
    GrangerCausalityResult,
    LikelihoodRatioResult,
    PairedTestResult,
)


class HypothesisReportBuilder:
    """Persist hypothesis testing outcomes for research documentation."""

    def __init__(self, output_dir: str = "outputs/hypothesis") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_reports(
        self,
        *,
        paired_results: Iterable[PairedTestResult],
        granger_results: Iterable[GrangerCausalityResult],
        lr_results: Iterable[LikelihoodRatioResult],
        metadata: Optional[Mapping[str, str]] = None,
        base_filename: Optional[str] = None,
    ) -> Mapping[str, Path]:
        """Create Markdown and HTML reports capturing hypothesis results."""

        paired_df = _paired_results_to_df(list(paired_results))
        granger_df = _granger_results_to_df(list(granger_results))
        lr_df = _likelihood_results_to_df(list(lr_results))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = base_filename or f"hypothesis_analysis_{timestamp}"

        markdown_path = self.output_dir / f"{filename}.md"
        html_path = self.output_dir / f"{filename}.html"

        markdown_path.write_text(
            _build_markdown(paired_df, granger_df, lr_df, metadata),
            encoding="utf-8",
        )
        html_path.write_text(
            _build_html(paired_df, granger_df, lr_df, metadata),
            encoding="utf-8",
        )

        return {"markdown": markdown_path, "html": html_path}


def _paired_results_to_df(results: List[PairedTestResult]) -> pd.DataFrame:
    records = []
    for res in results:
        record = {
            "Test": res.test_name,
            "Metric": res.metric_name,
            "Statistic": res.statistic,
            "P-Value": res.p_value,
            "Alpha": res.alpha,
            "Reject Null": res.reject_null,
        }
        if res.effect_size is not None:
            record["Effect Size"] = res.effect_size
        for key, value in res.details.items():
            record[_beautify_key(key)] = value
        records.append(record)
    return pd.DataFrame(records)


def _granger_results_to_df(results: List[GrangerCausalityResult]) -> pd.DataFrame:
    records = []
    for res in results:
        for lag_result in res.results:
            records.append({
                "Direction": res.direction,
                "Lag": lag_result.lag,
                "F-Statistic": lag_result.f_statistic,
                "F P-Value": lag_result.f_pvalue,
                "Chi2 Statistic": lag_result.chi2_statistic,
                "Chi2 P-Value": lag_result.chi2_pvalue,
            })
    return pd.DataFrame(records)


def _likelihood_results_to_df(results: List[LikelihoodRatioResult]) -> pd.DataFrame:
    records = []
    for res in results:
        record = {
            "Model": res.model_name,
            "Null LogLik": res.null_loglike,
            "Alt LogLik": res.alt_loglike,
            "LR Statistic": res.lr_statistic,
            "Degrees of Freedom": res.degrees_freedom,
            "P-Value": res.p_value,
            "Alpha": res.alpha,
            "Reject Null": res.reject_null,
        }
        for key, value in res.details.items():
            record[_beautify_key(key)] = value
        records.append(record)
    return pd.DataFrame(records)


def _build_markdown(
    paired_df: pd.DataFrame,
    granger_df: pd.DataFrame,
    lr_df: pd.DataFrame,
    metadata: Optional[Mapping[str, str]],
) -> str:
    lines = ["# Hypothesis Testing Summary", ""]
    if metadata:
        lines.append("## Context")
        for key, value in metadata.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")

    lines.extend(["## Paired Model Comparisons", _dataframe_to_markdown(paired_df), ""])
    lines.extend(["## Granger Causality", _dataframe_to_markdown(granger_df), ""])
    lines.extend(["## Likelihood Ratio Tests", _dataframe_to_markdown(lr_df), ""])

    return "\n".join(lines).strip() + "\n"


def _build_html(
    paired_df: pd.DataFrame,
    granger_df: pd.DataFrame,
    lr_df: pd.DataFrame,
    metadata: Optional[Mapping[str, str]],
) -> str:
    def _frame_to_html(df: pd.DataFrame) -> str:
        if df.empty:
            return "<p>No results available.</p>"
        return df.to_html(index=False, float_format="{:.4f}".format, border=0)

    meta_html = ""
    if metadata:
        rows = "".join(
            f"<li><strong>{key}</strong>: {value}</li>" for key, value in metadata.items()
        )
        meta_html = f"<ul>{rows}</ul>"

    return f"""
<html>
<head>
    <title>Hypothesis Testing Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 2rem; }}
        table {{ border-collapse: collapse; margin-bottom: 1.5rem; }}
        th, td {{ padding: 0.5rem 0.75rem; text-align: center; }}
        th {{ background-color: #f0f0f0; }}
    </style>
</head>
<body>
    <h1>Hypothesis Testing Summary</h1>
    {('<section><h2>Context</h2>' + meta_html + '</section>') if metadata else ''}
    <section>
        <h2>Paired Model Comparisons</h2>
        {_frame_to_html(paired_df)}
    </section>
    <section>
        <h2>Granger Causality</h2>
        {_frame_to_html(granger_df)}
    </section>
    <section>
        <h2>Likelihood Ratio Tests</h2>
        {_frame_to_html(lr_df)}
    </section>
</body>
</html>
"""


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No results available."
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        values = [
            _format_value(row[col])
            for col in columns
        ]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _beautify_key(key: str) -> str:
    return key.replace("_", " ").title()


def _format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


__all__ = ["HypothesisReportBuilder"]
