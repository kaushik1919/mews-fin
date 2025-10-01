"""Run robustness evaluations for MEWS under dataset perturbations."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from src.metrics import compute_cews_score
from src.ml_models import RiskPredictor

from .adversarial import AdversarialNoiseTester, NoiseReport
from .auditors import BiasComparison, BiasSummary, SentimentBiasAuditor
from .simulators import DelayReport, DelaySimulator

LOGGER = logging.getLogger(__name__)


@dataclass
class PerturbationConfig:
    name: str
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerturbationOutcome:
    name: str
    kind: str
    metrics: Dict[str, Any]
    deltas: Dict[str, float]
    details: Dict[str, Any]


@dataclass
class RobustnessReport:
    baseline_metrics: Dict[str, Any]
    perturbations: List[PerturbationOutcome]
    bias_report: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "baseline_metrics": self.baseline_metrics,
            "perturbations": [
                {
                    "name": p.name,
                    "kind": p.kind,
                    "metrics": p.metrics,
                    "deltas": p.deltas,
                    "details": p.details,
                }
                for p in self.perturbations
            ],
        }
        if self.bias_report is not None:
            payload["bias_report"] = self.bias_report
        return payload


class RobustnessEvaluator:
    """High level orchestrator for robustness perturbation experiments."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        target_col: str = "Risk_Label",
        feature_groups: Optional[Dict[str, List[str]]] = None,
        symbol_col: str = "Symbol",
        date_col: str = "Date",
        output_root: Path | str = Path("outputs/robustness"),
    ) -> None:
        if dataset.empty:
            raise ValueError("Dataset must contain rows for robustness evaluation")

        self.dataset = dataset
        self.target_col = target_col
        self.feature_groups = feature_groups
        self.symbol_col = symbol_col
        self.date_col = date_col
        self.output_root = Path(output_root)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.output_root / f"robustness_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = LOGGER

    def run(
        self,
        perturbations: Optional[Sequence[PerturbationConfig]] = None,
        auditor: Optional[SentimentBiasAuditor] = None,
    ) -> RobustnessReport:
        self.logger.info("Running baseline model on original dataset")
        baseline_metrics = self._evaluate_dataframe(self.dataset, label="baseline")

        perturbation_configs = list(perturbations or self._default_perturbations())
        perturbation_results: List[PerturbationOutcome] = []

        for config in perturbation_configs:
            self.logger.info("Applying perturbation '%s' (%s)", config.name, config.kind)
            perturbed_df, details = self._apply_perturbation(config)
            metrics = self._evaluate_dataframe(perturbed_df, label=config.name)
            deltas = self._compute_metric_deltas(baseline_metrics, metrics)
            perturbation_results.append(
                PerturbationOutcome(
                    name=config.name,
                    kind=config.kind,
                    metrics=metrics,
                    deltas=deltas,
                    details=details,
                )
            )

        bias_payload: Optional[Dict[str, Any]] = None
        if auditor is not None:
            self.logger.info("Running sentiment bias audit")
            bias_results = auditor.audit(self.dataset, group_col=self.symbol_col if self.symbol_col in self.dataset.columns else None)
            bias_payload = self._serialize_bias_report(bias_results)

        report = RobustnessReport(
            baseline_metrics=baseline_metrics,
            perturbations=perturbation_results,
            bias_report=bias_payload,
        )

        output_path = self.output_dir / "robustness_report.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2, default=self._json_default)
        self.logger.info("Robustness report written to %s", output_path)

        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _default_perturbations(self) -> Iterable[PerturbationConfig]:
        return [
            PerturbationConfig(name="noise_5pct", kind="noise", params={"noise_level": 0.05}),
            PerturbationConfig(name="noise_10pct", kind="noise", params={"noise_level": 0.10}),
            PerturbationConfig(name="delay_1d", kind="delay", params={"delay_days": 1}),
        ]

    def _apply_perturbation(self, config: PerturbationConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if config.kind == "noise":
            tester = AdversarialNoiseTester()
            perturbed, report = tester.apply(self.dataset, **config.params)
            details = asdict(report)
            return perturbed, details
        if config.kind == "delay":
            simulator = DelaySimulator(date_col=self.date_col)
            perturbed, report = simulator.apply(self.dataset, **config.params)
            details = asdict(report)
            return perturbed, details
        raise ValueError(f"Unsupported perturbation kind: {config.kind}")

    def _evaluate_dataframe(self, df: pd.DataFrame, label: str) -> Dict[str, Any]:
        predictor = RiskPredictor()
        features_df, target, feature_names = predictor.prepare_modeling_data(
            df,
            feature_groups=self.feature_groups,
            target_col=self.target_col,
        )

        if features_df.empty or target.size == 0:
            raise ValueError("Prepared features or target are empty; cannot evaluate")

        metrics = predictor.train_models(features_df, target, feature_names)

        metadata = predictor.training_metadata
        meta_df = metadata.copy() if metadata is not None else pd.DataFrame(index=features_df.index)

        if self.date_col in df.columns and self.date_col not in meta_df.columns:
            meta_df[self.date_col] = pd.to_datetime(df.loc[features_df.index, self.date_col])
        if self.symbol_col in df.columns and self.symbol_col not in meta_df.columns:
            meta_df[self.symbol_col] = df.loc[features_df.index, self.symbol_col]

        _, probabilities = predictor.predict_risk(features_df, model_type="ensemble", metadata=meta_df)

        evaluation_frame = meta_df.copy()
        evaluation_frame["Risk_Probability"] = probabilities
        evaluation_frame[self.target_col] = target

        cews = compute_cews_score(
            evaluation_frame,
            probability_col="Risk_Probability",
            label_col=self.target_col,
            date_col=self.date_col if self.date_col in evaluation_frame.columns else "Date",
            symbol_col=self.symbol_col if self.symbol_col in evaluation_frame.columns else None,
        )

        ensemble_metrics = metrics.get("ensemble", {})
        ensemble_metrics["cews_score"] = cews.score
        ensemble_metrics["cews_early_detection"] = cews.early_detection_reward
        ensemble_metrics["cews_false_alarm_penalty"] = cews.false_alarm_penalty

        label_distribution = self._label_distribution(target)

        return {
            "label": label,
            "ensemble": self._extract_metric_summary(ensemble_metrics),
            "all_models": self._extract_all_metrics(metrics),
            "cews": {
                "score": cews.score,
                "early_detection_reward": cews.early_detection_reward,
                "false_alarm_penalty": cews.false_alarm_penalty,
            },
            "label_distribution": label_distribution,
        }

    def _extract_metric_summary(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        summary: Dict[str, float] = {}
        for key in ["auc_score", "test_accuracy", "fbeta_score", "cews_score", "cews_early_detection", "cews_false_alarm_penalty"]:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                summary[key] = float(value)
        return summary

    def _extract_all_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        payload: Dict[str, Dict[str, float]] = {}
        for model_name, model_metrics in metrics.items():
            if not isinstance(model_metrics, dict):
                continue
            payload[model_name] = {}
            for key, value in model_metrics.items():
                if isinstance(value, (int, float)):
                    payload[model_name][key] = float(value)
        return payload

    def _label_distribution(self, target: Any) -> Dict[str, float]:
        series = pd.Series(target)
        counts = series.value_counts(normalize=True)
        return {str(k): float(v) for k, v in counts.items()}

    def _compute_metric_deltas(
        self,
        baseline: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> Dict[str, float]:
        baseline_metrics = baseline.get("ensemble", {})
        candidate_metrics = candidate.get("ensemble", {})
        deltas: Dict[str, float] = {}
        for key in set(baseline_metrics.keys()) | set(candidate_metrics.keys()):
            base_value = baseline_metrics.get(key)
            cand_value = candidate_metrics.get(key)
            if isinstance(base_value, (int, float)) and isinstance(cand_value, (int, float)):
                deltas[key] = float(cand_value - base_value)
        return deltas

    def _serialize_bias_report(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        sources = {
            name: {
                "aggregated_mean": summary.aggregated_mean,
                "aggregated_skew": summary.aggregated_skew,
                "statistics": [asdict(stat) for stat in summary.statistics],
            }
            for name, summary in payload.get("sources", {}).items()
        }
        comparisons = [asdict(comp) for comp in payload.get("comparisons", [])]
        group_entries = [
            {key: (float(value) if isinstance(value, (int, float)) else value) for key, value in record.items()}
            for record in payload.get("group_summaries", [])
        ]
        return {
            "sources": sources,
            "comparisons": comparisons,
            "group_summaries": group_entries,
        }

    def _json_default(self, obj: Any) -> Any:
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, (BiasSummary, BiasComparison, NoiseReport, DelayReport)):
            return asdict(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_dataset(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def run_cli(args: Optional[Sequence[str]] = None) -> RobustnessReport:
    import argparse

    parser = argparse.ArgumentParser(description="Run MEWS robustness evaluations.")
    parser.add_argument("dataset", type=str, help="Path to integrated dataset")
    parser.add_argument(
        "--feature-groups",
        type=str,
        default=None,
        help="Optional JSON/YAML file describing feature groups",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/robustness",
        help="Directory where robustness reports will be stored",
    )
    parser.add_argument(
        "--skip-bias",
        action="store_true",
        help="Skip sentiment bias auditing",
    )

    parsed = parser.parse_args(args=args)

    dataset = load_dataset(parsed.dataset)
    feature_groups = None
    if parsed.feature_groups:
        fg_path = Path(parsed.feature_groups)
        if not fg_path.exists():
            raise FileNotFoundError(fg_path)
        if fg_path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise ImportError("PyYAML is required to load feature group YAML files") from exc
            with fg_path.open("r", encoding="utf-8") as handle:
                feature_groups = yaml.safe_load(handle)
        else:
            with fg_path.open("r", encoding="utf-8") as handle:
                feature_groups = json.load(handle)
        if not isinstance(feature_groups, dict):
            raise ValueError("Feature group specification must be a mapping")

    evaluator = RobustnessEvaluator(
        dataset=dataset,
        feature_groups=feature_groups,
        output_root=parsed.output,
    )

    auditor = None if parsed.skip_bias else SentimentBiasAuditor(
        finbert_columns=[col for col in dataset.columns if col.lower().endswith("_sentiment")],
        vader_columns=[
            col
            for col in dataset.columns
            if "sentiment" in col.lower() and col.lower().endswith("compound")
        ],
    )

    return evaluator.run(auditor=auditor)


__all__ = [
    "PerturbationConfig",
    "RobustnessEvaluator",
    "run_cli",
    "load_dataset",
]
