"""Experiment manager for orchestrating MEWS baselines and model comparisons."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Type

import pandas as pd

from src.baselines import (
    BaselineResult,
    BaseBaseline,
    GARCHBaseline,
    LSTMBaseline,
    ValueAtRiskBaseline,
)

try:  # pragma: no cover - optional dependency
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from typing import TypedDict
except ImportError:  # pragma: no cover
    TypedDict = Dict[str, Any]  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration payload describing an experiment run."""

    name: str = "baseline_experiment"
    data_path: Optional[str] = None
    baselines: List[Dict[str, Any]] = field(default_factory=list)
    mews: Dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    ablations: List[Dict[str, Any]] = field(default_factory=list)
    output_dir: str = "outputs/experiments"
    date_col: str = "Date"
    symbol_col: str = "Symbol"
    target_col: str = "Risk_Label"
    returns_col: str = "Returns"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ExperimentConfig":
        values = dict(payload)
        return cls(**values)

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(path)

        if path_obj.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise ImportError("PyYAML is required to load YAML experiment configs")
            with path_obj.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle)
        else:
            with path_obj.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("Experiment configuration must be a mapping/dict")
        return cls.from_mapping(payload)


def _slugify(value: str) -> str:
    lowered = value.lower()
    cleaned = [ch if ch.isalnum() else "_" for ch in lowered]
    slug = "".join(cleaned)
    while "__" in slug:
        slug = slug.replace("__", "_")
    slug = slug.strip("_")
    return slug or "experiment"


class ExperimentManager:
    """Coordinate baseline runs, ablations, and MEWS model comparisons."""

    BASELINE_REGISTRY: Dict[str, Type[BaseBaseline]] = {
        "garch": GARCHBaseline,
        "garch_1_1": GARCHBaseline,
        "value_at_risk": ValueAtRiskBaseline,
        "var": ValueAtRiskBaseline,
        "lstm": LSTMBaseline,
        "lstm_risk_predictor": LSTMBaseline,
    }

    def __init__(
        self,
        config: ExperimentConfig,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> None:
        self.config = config
        self._provided_df = dataframe.copy() if dataframe is not None else None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = (
            Path(config.output_dir) / f"{timestamp}_{_slugify(config.name)}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = LOGGER
        self.logger.info("Experiment outputs will be stored in %s", self.output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """Execute the configured experiment and return a structured summary."""

        base_df = self._load_data()
        runs: List[Dict[str, Any]] = []

        baseline_run = self._execute_run(
            label="baseline",
            df=base_df,
            baseline_overrides=None,
            mews_overrides=None,
        )
        runs.append(baseline_run)

        for ablation in self.config.ablations:
            name = str(ablation.get("name", "ablation"))
            ablation_df = self._apply_ablation(base_df, ablation)
            baseline_overrides = (
                ablation.get("baselines") if isinstance(ablation, Mapping) else None
            )
            mews_overrides = (
                ablation.get("mews") if isinstance(ablation, Mapping) else None
            )
            ablation_run = self._execute_run(
                label=name,
                df=ablation_df,
                baseline_overrides=baseline_overrides,
                mews_overrides=mews_overrides,
            )
            runs.append(ablation_run)

        summary = {
            "experiment": self.config.name,
            "output_dir": str(self.output_dir),
            "runs": runs,
        }
        self._write_json(summary, self.output_dir / "summary.json")
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_data(self) -> pd.DataFrame:
        if self._provided_df is not None:
            return self._provided_df.copy()

        if not self.config.data_path:
            raise ValueError(
                "Experiment config missing 'data_path' and no dataframe provided"
            )

        path = Path(self.config.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        if path.suffix.lower() in {".csv"}:
            df = pd.read_csv(path)
        elif path.suffix.lower() in {".parquet"}:
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported dataset format: {path.suffix}")
        return df

    def _execute_run(
        self,
        label: str,
        df: pd.DataFrame,
        baseline_overrides: Optional[Mapping[str, Any]],
        mews_overrides: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        self.logger.info("\n=== Running experiment segment: %s ===", label)
        results: Dict[str, Any] = {"label": label}

        baseline_configs = self.config.baselines or self._default_baselines()
        baseline_results: List[BaselineResult] = []

        for entry in baseline_configs:
            try:
                baseline = self._instantiate_baseline(entry, baseline_overrides)
            except Exception as exc:
                self.logger.error("Failed to instantiate baseline %s: %s", entry, exc)
                continue

            try:
                result = baseline.run(
                    df,
                    symbol_col=self.config.symbol_col,
                    date_col=self.config.date_col,
                )
            except Exception as exc:
                self.logger.error("Baseline %s failed: %s", baseline.name, exc)
                continue

            baseline_results.append(result)
            self._persist_baseline(label, result)

        if baseline_results:
            combined = self._combine_predictions(baseline_results)
            combined_path = self._write_dataframe(
                combined,
                self.output_dir / f"{label}_baselines.csv",
            )
            results["combined_baselines"] = combined_path
            results["baselines"] = [
                self._result_summary(res) for res in baseline_results
            ]
        else:
            results["baselines"] = []

        mews_cfg = self._merge_dict(self.config.mews, mews_overrides)
        if mews_cfg.get("enabled", False):
            mews_result = self._run_mews(df, label, mews_cfg)
            if mews_result:
                results["mews_model"] = mews_result
        return results

    def _instantiate_baseline(
        self,
        entry: Mapping[str, Any],
        overrides: Optional[Mapping[str, Any]],
    ) -> BaseBaseline:
        kind = str(entry.get("type", "")).lower()
        if not kind:
            raise ValueError("Baseline entry missing 'type'")

        params = dict(entry.get("params", {}))
        if overrides and kind in overrides:
            override_payload = overrides.get(kind, {})
            if isinstance(override_payload, Mapping):
                params.update(override_payload)

        factory = self.BASELINE_REGISTRY.get(kind)
        if factory is None:
            raise ValueError(f"Unknown baseline type: {kind}")
        params.setdefault("returns_col", self.config.returns_col)
        if factory is LSTMBaseline:
            params.setdefault("target_col", self.config.target_col)
        baseline: BaseBaseline = factory(**params)
        return baseline

    def _default_baselines(self) -> List[Mapping[str, Any]]:
        return [
            {"type": "garch"},
            {"type": "value_at_risk"},
            {"type": "lstm"},
        ]

    def _persist_baseline(self, label: str, result: BaselineResult) -> None:
        preds_path = self.output_dir / f"{label}_{result.name}.csv"
        meta_path = self.output_dir / f"{label}_{result.name}_metadata.json"
        self._write_dataframe(result.predictions, preds_path)
        self._write_json(result.metadata, meta_path)

    def _combine_predictions(self, results: Iterable[BaselineResult]) -> pd.DataFrame:
        merged: Optional[pd.DataFrame] = None
        key_cols = [self.config.symbol_col, self.config.date_col]
        for result in results:
            preds = result.predictions.copy()
            if merged is None:
                merged = preds
                continue
            merged = merged.merge(
                preds, on=key_cols, how="outer", suffixes=("", f"_{result.name}")
            )
        return merged if merged is not None else pd.DataFrame(columns=key_cols)

    def _run_mews(
        self,
        df: pd.DataFrame,
        label: str,
        mews_cfg: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        try:
            from src.ml_models import RiskPredictor
        except ImportError as exc:  # pragma: no cover
            self.logger.error("Unable to import RiskPredictor: %s", exc)
            return None

        predictor = RiskPredictor()
        regime_cfg = (
            mews_cfg.get("regime_adaptive") if isinstance(mews_cfg, Mapping) else None
        )
        if isinstance(regime_cfg, Mapping):
            predictor.set_regime_adaptive_options(dict(regime_cfg))
        elif isinstance(regime_cfg, bool):
            predictor.set_regime_adaptive_options({"enabled": regime_cfg})
        else:
            predictor.set_regime_adaptive_options(None)
        feature_groups = self._load_feature_groups(mews_cfg.get("feature_groups"))

        try:
            features_df, target, feature_names = predictor.prepare_modeling_data(
                df,
                feature_groups=feature_groups,
                target_col=mews_cfg.get("target_col", self.config.target_col),
            )
        except Exception as exc:
            self.logger.error("Failed to prepare data for MEWS model: %s", exc)
            return None

        if features_df.empty or len(target) == 0:
            self.logger.warning("MEWS model skipped: no features/targets available")
            return None

        test_size = float(mews_cfg.get("test_size", 0.2))
        random_state = int(mews_cfg.get("random_state", 42))

        try:
            metrics = predictor.train_models(
                features_df,
                target,
                feature_names,
                test_size=test_size,
                random_state=random_state,
            )
        except Exception as exc:
            self.logger.error("MEWS model training failed: %s", exc)
            return None

        metadata_df = predictor.training_metadata
        if metadata_df is None or metadata_df.empty:
            metadata_df = df.loc[
                features_df.index, [self.config.symbol_col, self.config.date_col]
            ]
        else:
            metadata_df = metadata_df[[self.config.symbol_col, self.config.date_col]]

        try:
            predictions, probabilities = predictor.predict_risk(
                features_df,
                model_type=mews_cfg.get("model_type", "ensemble"),
                metadata=metadata_df,
            )
        except Exception as exc:
            self.logger.error("MEWS prediction failed: %s", exc)
            return None

        prediction_df = metadata_df.copy()
        prediction_df["mews_risk_probability"] = probabilities
        prediction_df["mews_prediction"] = predictions

        preds_path = self._write_dataframe(
            prediction_df,
            self.output_dir / f"{label}_mews_predictions.csv",
        )
        metrics_path = self._write_json(
            self._to_serializable(metrics),
            self.output_dir / f"{label}_mews_metrics.json",
        )

        regime_payload: Optional[Dict[str, Any]] = None
        if predictor.dynamic_ensemble is not None:
            regime_summary = predictor.dynamic_ensemble.to_json()
            regime_path = self._write_json(
                self._to_serializable(regime_summary),
                self.output_dir / f"{label}_mews_regime_weights.json",
            )
            regime_payload = {
                "config": predictor.regime_options,
                "summary": regime_summary,
                "artifact": regime_path,
                "meta_model_enabled": predictor.use_regime_meta_model,
            }

        payload = {
            "predictions": preds_path,
            "metrics": metrics_path,
            "thresholds": predictor.thresholds,
        }
        if regime_payload:
            payload["regime_adaptive"] = regime_payload
        return payload

    def _load_feature_groups(self, spec: Any) -> Optional[Dict[str, List[str]]]:
        if spec is None:
            return None
        if isinstance(spec, Mapping):
            return {str(k): list(v) for k, v in spec.items()}
        if isinstance(spec, str):
            path = Path(spec)
            if not path.exists():
                raise FileNotFoundError(f"Feature group spec not found: {spec}")
            with path.open("r", encoding="utf-8") as handle:
                if path.suffix.lower() in {".json"}:
                    data = json.load(handle)
                elif path.suffix.lower() in {".yaml", ".yml"}:
                    if yaml is None:
                        raise ImportError("PyYAML required to load YAML feature groups")
                    data = yaml.safe_load(handle)
                else:
                    raise ValueError("Unsupported feature group format")
            if not isinstance(data, Mapping):
                raise ValueError("Feature group specification must be a mapping")
            return {str(k): list(v) for k, v in data.items()}
        raise TypeError("feature_groups must be a mapping or path to mapping")

    def _apply_ablation(
        self, df: pd.DataFrame, ablation: Mapping[str, Any]
    ) -> pd.DataFrame:
        working = df.copy()
        drop_cols = ablation.get("drop_columns") or []
        if drop_cols:
            working = working.drop(
                columns=[col for col in drop_cols if col in working.columns]
            )

        keep_cols = ablation.get("keep_columns")
        if keep_cols:
            columns = [col for col in keep_cols if col in working.columns]
            working = working.loc[:, columns]

        filters = ablation.get("filters") or {}
        for column, allowed in filters.items():
            if column in working.columns:
                working = working[working[column].isin(allowed)]

        return working

    def _merge_dict(
        self,
        base: Mapping[str, Any],
        overrides: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        merged = dict(base or {})
        if overrides:
            merged.update({k: v for k, v in overrides.items() if v is not None})
        return merged

    def _result_summary(self, result: BaselineResult) -> Dict[str, Any]:
        summary = {
            "name": result.name,
            "rows": len(result.predictions),
        }
        numeric_cols = result.predictions.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            summary[f"mean_{col}"] = float(result.predictions[col].mean())

        summary.update(
            {
                f"meta_{k}": v
                for k, v in result.metadata.items()
                if not isinstance(v, dict)
            }
        )
        return summary

    def _write_dataframe(self, df: pd.DataFrame, path: Path) -> str:
        df.to_csv(path, index=False)
        return str(path)

    def _write_json(self, payload: Any, path: Path) -> str:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=self._json_default)
        return str(path)

    def _json_default(self, obj: Any) -> Any:  # pragma: no cover - JSON helper
        try:
            if hasattr(obj, "tolist"):
                return obj.tolist()
        except Exception:
            pass
        if isinstance(obj, set):
            return list(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _to_serializable(self, payload: Any) -> Any:
        if isinstance(payload, Mapping):
            return {k: self._to_serializable(v) for k, v in payload.items()}
        if isinstance(payload, list):
            return [self._to_serializable(v) for v in payload]
        if hasattr(payload, "tolist"):
            return payload.tolist()
        return payload


__all__ = ["ExperimentConfig", "ExperimentManager"]
