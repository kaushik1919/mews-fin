"""Research reporting utilities for MEWS enhancements."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


class ResearchReportBuilder:
    def __init__(self, output_dir: str = "outputs/research") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_markdown(
        self,
        evaluation_results: Dict[str, Any],
        hypothesis_results: Dict[str, Any],
        robustness_results: Dict[str, Any],
        filename: str = "research_overview.md",
    ) -> Path:
        lines = [
            "# Market Risk Early Warning System – Research Addendum",
            "",
            "## 1. Evaluation Summary",
        ]
        for section, metrics in evaluation_results.items():
            lines.append(f"### {section.replace('_', ' ').title()}")
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, dict):
                        lines.append(f"- **{key}**: {json.dumps(value, indent=2)}")
                    else:
                        lines.append(f"- **{key}**: {value}")
            lines.append("")

        lines.extend([
            "## 2. Hypothesis Tests",
            json.dumps(hypothesis_results, indent=2),
            "",
            "## 3. Robustness Checks",
            json.dumps(robustness_results, indent=2),
        ])

        output_path = self.output_dir / filename
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    def build_html(
        self,
        evaluation_results: Dict[str, Any],
        hypothesis_results: Dict[str, Any],
        robustness_results: Dict[str, Any],
        filename: str = "research_overview.html",
    ) -> Path:
        df_eval = pd.json_normalize(evaluation_results, sep=".")
        df_hyp = pd.json_normalize(hypothesis_results, sep=".")
        df_robust = pd.json_normalize(robustness_results, sep=".")

        html_content = """
        <html>
        <head><title>MEWS Research Addendum</title></head>
        <body>
        <h1>Market Risk Early Warning System – Research Addendum</h1>
        <h2>Evaluation Summary</h2>
        {eval_table}
        <h2>Hypothesis Tests</h2>
        {hyp_table}
        <h2>Robustness Checks</h2>
        {robust_table}
        </body>
        </html>
        """

        html = html_content.format(
            eval_table=df_eval.to_html(index=False, border=0, justify="center"),
            hyp_table=df_hyp.to_html(index=False, border=0, justify="center"),
            robust_table=df_robust.to_html(index=False, border=0, justify="center"),
        )

        output_path = self.output_dir / filename
        output_path.write_text(html, encoding="utf-8")
        return output_path


__all__ = ["ResearchReportBuilder"]
