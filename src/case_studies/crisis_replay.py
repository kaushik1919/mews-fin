"""Command-line entry point to replay predefined crisis case studies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    # Enable execution via ``python src/case_studies/crisis_replay.py``
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from case_studies.runner import CaseStudyRunner  # type: ignore
    from case_studies.scenarios import CaseStudyScenario, PREDEFINED_CASE_STUDIES  # type: ignore
else:
    from .runner import CaseStudyRunner
    from .scenarios import CaseStudyScenario, PREDEFINED_CASE_STUDIES


def _scenario_by_slug() -> dict[str, CaseStudyScenario]:
    return {scenario.slug: scenario for scenario in PREDEFINED_CASE_STUDIES}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay financial crisis case studies with MEWS risk outputs."
    )
    parser.add_argument(
        "--scenario",
        "-s",
        dest="scenarios",
        nargs="*",
        choices=[scenario.slug for scenario in PREDEFINED_CASE_STUDIES],
        help="Specific scenario slug(s) to run (default: all).",
    )
    parser.add_argument(
        "--data",
        default="data/integrated_dataset.csv",
        help="Path to integrated dataset CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--predictions",
        default="outputs/risk_predictions.csv",
        help="Path to risk predictions CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--features",
        default="data/feature_groups.json",
        help="Path to feature group mapping JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default="outputs/case_studies",
        help="Directory for exported figures and reports (default: %(default)s).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a JSON summary of generated artifacts to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    available = _scenario_by_slug()
    slugs = args.scenarios or list(available.keys())

    runner = CaseStudyRunner(
        data_path=args.data,
        predictions_path=args.predictions,
        feature_groups_path=args.features,
        output_dir=args.output,
    )

    results = []
    for slug in slugs:
        scenario = available[slug]
        result = runner.run_case_study(scenario)
        results.append(result)

    if args.summary:
        summary = [
            {
                "scenario": res.scenario.slug,
                "plot": str(res.plot_path) if res.plot_path else None,
                "report": str(res.report_path) if res.report_path else None,
                "warnings": res.warning_events,
                "downturns": res.downturn_events,
                "combined": res.combined_events,
                "top_features": res.top_features,
            }
            for res in results
        ]
        print(json.dumps(summary, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
