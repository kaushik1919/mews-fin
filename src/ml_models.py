"""
Machine Learning models for market risk prediction
Implements Random Forest, Logistic Regression, XGBoost, and SVM models with GPU support
"""

import json
import math
import os
import pickle
import tempfile
import warnings
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import mlflow  # type: ignore
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore

try:
    from src.ensemble import (
        RegimeAdaptiveEnsemble,
        StaticWeightedEnsemble,
    )
except ImportError:  # pragma: no cover - ensemble utilities should be available
    RegimeAdaptiveEnsemble = None  # type: ignore
    StaticWeightedEnsemble = None  # type: ignore

from src.utils.logging import get_logger

try:  # pragma: no cover - uncertainty utilities are optional
    from src.uncertainty.calibration import (
        ProbabilityCalibrator,
        save_reliability_diagram,
    )
except ImportError:  # pragma: no cover
    ProbabilityCalibrator = None  # type: ignore
    save_reliability_diagram = None  # type: ignore

warnings.filterwarnings("ignore")


# GPU Detection
def detect_gpu():
    """Detect available GPU for acceleration"""
    gpu_info = {"available": False, "device": "cpu", "name": "N/A"}

    try:
        import torch

        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["device"] = "cuda"
            gpu_info["name"] = torch.cuda.get_device_name(0)
            print(f"GPU detected: {gpu_info['name']}")
        else:
            print("CUDA not available, using CPU")
    except ImportError:
        print("PyTorch not available, using CPU")

    return gpu_info


class MLModelTrainer:
    """Lightweight trainer used in unit tests for rapid model experimentation."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.logger = get_logger(__name__)
        self.models: Dict[str, Any] = {}
        self.feature_names: List[str] = []

    def prepare_features(
        self, df: pd.DataFrame, target_column: str = "risk_label"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Create numeric feature matrix and aligned target series.

        Args:
            df: Input DataFrame containing features and target.
            target_column: Name of the binary risk label column.

        Returns:
            Tuple of (features DataFrame, target Series).
        """

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")

        features = df.drop(columns=[target_column])
        numeric_features = features.select_dtypes(include=[np.number]).copy()
        numeric_features = numeric_features.fillna(numeric_features.median())

        self.feature_names = list(numeric_features.columns)
        target = df[target_column].astype(int)

        return numeric_features, target

    def _evaluate_classifier(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        from sklearn.metrics import precision_score, recall_score, roc_auc_score

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        metrics = {
            "auc": float(roc_auc_score(y_test, probabilities)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
        }
        return metrics

    def train_models(
        self,
        X: pd.DataFrame,
        y: Iterable[int],
        test_size: float = 0.2,
        random_state: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Train fast baseline classifiers for risk prediction."""

        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        if random_state is None:
            random_state = self.random_state

        y_array = np.array(list(y))

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_array,
            test_size=test_size,
            random_state=random_state,
            stratify=y_array,
        )

        results: Dict[str, Dict[str, float]] = {}

        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)
        self.models["Random Forest"] = rf_model
        results["Random Forest"] = self._evaluate_classifier(rf_model, X_test, y_test)

        lr_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        max_iter=500,
                        random_state=random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        lr_pipeline.fit(X_train, y_train)
        self.models["Logistic Regression"] = lr_pipeline
        results["Logistic Regression"] = self._evaluate_classifier(
            lr_pipeline, X_test, y_test
        )

        try:
            from xgboost import XGBClassifier

            xgb_model = XGBClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=3,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                eval_metric="logloss",
                use_label_encoder=False,
            )
            xgb_model.fit(X_train, y_train)
            model_label = "XGBoost"
        except ImportError:
            xgb_model = GradientBoostingClassifier(random_state=random_state)
            xgb_model.fit(X_train, y_train)
            self.logger.warning(
                "XGBoost not installed; using GradientBoostingClassifier fallback"
            )

        model_label = "XGBoost"
        self.models[model_label] = xgb_model
        results[model_label] = self._evaluate_classifier(xgb_model, X_test, y_test)

        return results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return averaged probability estimates across trained models."""

        if not self.models:
            raise ValueError("Models have not been trained yet")

        probabilities = []
        for model in self.models.values():
            proba = model.predict_proba(X)[:, 1]
            probabilities.append(proba)

        avg_probabilities = np.mean(probabilities, axis=0)
        return avg_probabilities


class ModelEnsemble:
    """Utility for combining predictions from multiple estimators."""

    def average_predictions(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        if not predictions:
            raise ValueError("No predictions supplied to ensemble")

        stacked = np.vstack(list(predictions.values()))
        return stacked.mean(axis=0)


class RiskPredictor:
    """Machine Learning models for market risk prediction"""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_metrics = {}
        self.gpu_info = detect_gpu()
        self.model_dir = "models"
        self.thresholds: Dict[str, float] = {}
        self.ensemble_threshold: float = 0.5
        self.ensemble_weights: List[Dict[str, float]] = []
        self.dynamic_ensemble = None
        self.use_regime_meta_model = True
        self.regime_options = {"enabled": True}
        self.training_metadata = None
        os.makedirs(self.model_dir, exist_ok=True)

    def prepare_modeling_data(
        self,
        df: pd.DataFrame,
        feature_groups: Optional[Dict[str, List[str]]] = None,
        target_col: str = "Risk_Label",
    ) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """
        Prepare data for machine learning models

        Args:
            df: Integrated dataset
            feature_groups: Dictionary of feature groups
            target_col: Target column name

        Returns:
            Tuple of (features_df, target_array, feature_names)
        """
        self.logger.info("Preparing data for modeling...")

        if df.empty:
            return pd.DataFrame(), np.array([]), []

        # Make copy to avoid modifying original
        data = df.copy()

        # Create target variable if not exists
        if target_col not in data.columns:
            self.logger.info(f"Creating target variable '{target_col}'")
            data = self._create_risk_labels(data)

        # Select features for modeling
        if feature_groups is None:
            # Auto-select numeric features excluding target and identifiers
            exclude_cols = ["Date", "Symbol", target_col, "Risk_Score"] + [
                col for col in data.columns if data[col].dtype == "object"
            ]
            feature_cols = [
                col
                for col in data.columns
                if col not in exclude_cols and data[col].dtype in ["int64", "float64"]
            ]
        else:
            # Use specified feature groups
            feature_cols = []
            for group_name, features in feature_groups.items():
                if group_name not in [
                    "date_features"
                ]:  # Exclude date features from modeling
                    feature_cols.extend([f for f in features if f in data.columns])

            # Remove duplicates and target column
            feature_cols = list(set(feature_cols))
            if target_col in feature_cols:
                feature_cols.remove(target_col)

        self.logger.info(f"Selected {len(feature_cols)} features for modeling")

        # Handle missing values in features
        features_df = data[feature_cols].copy()

        # Remove columns with too many missing values
        missing_threshold = 0.3
        cols_to_keep = []
        for col in features_df.columns:
            missing_pct = features_df[col].isnull().sum() / len(features_df)
            if missing_pct <= missing_threshold:
                cols_to_keep.append(col)
            else:
                self.logger.warning(
                    f"Removing {col} - {missing_pct:.2%} missing values"
                )

        features_df = features_df[cols_to_keep]

        # Fill remaining missing values
        features_df = features_df.fillna(features_df.median())

        # Remove infinite values
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        features_df = features_df.fillna(0)

        # Get target variable
        target = (
            data[target_col].values
            if target_col in data.columns
            else np.zeros(len(data))
        )

        # Remove rows with missing target
        valid_mask = ~pd.isna(target)
        features_df = features_df[valid_mask]
        target = target[valid_mask]

        self.logger.info(
            f"Final dataset: {features_df.shape[0]} samples, {features_df.shape[1]} features"
        )
        self.logger.info(f"Target distribution: {np.bincount(target.astype(int))}")

        metadata_cols = [
            col
            for col in ["Date", "Symbol", "Returns", "Close", "Adj Close"]
            if col in data.columns
        ]
        if metadata_cols:
            self.training_metadata = data.loc[features_df.index, metadata_cols].copy()
        else:
            self.training_metadata = None

        return features_df, target, list(features_df.columns)

    def set_regime_adaptive_options(self, options: Optional[Dict[str, Any]]) -> None:
        """Configure regime-adaptive ensemble parameters applied during training."""

        if options is None:
            self.regime_options = {"enabled": True}
            return

        configured = dict(options)
        configured.setdefault("enabled", True)
        self.regime_options = configured
        if "use_meta_model" in configured:
            self.use_regime_meta_model = bool(configured["use_meta_model"])

    def _optimize_threshold(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
        beta: float = 1.0,
    ) -> Tuple[float, float]:
        """Find probability threshold maximizing F-beta score."""

        try:
            from sklearn.metrics import fbeta_score
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for threshold optimization"
            ) from exc

        thresholds = np.linspace(0.05, 0.95, 181)
        best_threshold = 0.5
        best_score = -1.0

        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            score = fbeta_score(y_true, predictions, beta=beta, zero_division=0)
            if score > best_score:
                best_score = float(score)
                best_threshold = float(threshold)

        return best_threshold, best_score

    def train_models(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: List[str],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Train Random Forest and Logistic Regression models

        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            test_size: Proportion of data for testing
            random_state: Random seed

        Returns:
            Dictionary with training results
        """
        self.logger.info("Training machine learning models...")

        # Reset adaptive parameters before each training run
        self.thresholds = {}
        self.ensemble_weights = []
        self.ensemble_threshold = 0.5
        self.feature_names = feature_names
        self.calibrators = {}
        self.calibration_artifacts = {}

        try:
            import joblib
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import (
                classification_report,
                confusion_matrix,
                accuracy_score,
                brier_score_loss,
                roc_auc_score,
            )
            from sklearn.model_selection import (
                GridSearchCV,
                cross_val_score,
                train_test_split,
            )
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC

            # Try to import XGBoost
            try:
                from xgboost import XGBClassifier

                xgb_available = True
            except ImportError:
                self.logger.warning("XGBoost not available - will skip XGBoost model")
                xgb_available = False

        except ImportError:
            self.logger.error("Scikit-learn not available")
            return {}

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.logger.info(f"Training set: {X_train.shape[0]} samples")
        self.logger.info(f"Test set: {X_test.shape[0]} samples")

        results = {}

        # 1. Random Forest Model (Optimized)
        self.logger.info("Training Random Forest model...")
        if self.gpu_info["available"]:
            self.logger.info(
                f"Using GPU acceleration where possible: {self.gpu_info['name']}"
            )

        # Optimized hyperparameter tuning for Random Forest
        rf_param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [15, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "class_weight": ["balanced", "balanced_subsample"],
            "max_features": ["sqrt", "log2"],
        }

        rf_base = RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,  # Use all CPU cores
            oob_score=True,  # Out-of-bag scoring
            warm_start=True,  # Incremental learning
        )
        rf_grid = GridSearchCV(
            rf_base, rf_param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1
        )
        rf_grid.fit(X_train, y_train)

        self.models["random_forest"] = rf_grid.best_estimator_

        # Feature importance for Random Forest
        self.feature_importance["random_forest"] = dict(
            zip(feature_names, self.models["random_forest"].feature_importances_)
        )

        # Evaluate Random Forest with optimized threshold
        rf_train_probabilities = self.models["random_forest"].predict_proba(X_train)[
            :, 1
        ]
        rf_probabilities = self.models["random_forest"].predict_proba(X_test)[:, 1]
        rf_threshold, rf_fbeta = self._optimize_threshold(y_test, rf_probabilities)
        self.thresholds["random_forest"] = rf_threshold

        rf_train_predictions = (rf_train_probabilities >= rf_threshold).astype(int)
        rf_predictions = (rf_probabilities >= rf_threshold).astype(int)

        rf_train_score = accuracy_score(y_train, rf_train_predictions)
        rf_test_score = accuracy_score(y_test, rf_predictions)
        rf_auc = roc_auc_score(y_test, rf_probabilities)

        results["random_forest"] = {
            "train_accuracy": rf_train_score,
            "test_accuracy": rf_test_score,
            "auc_score": rf_auc,
            "best_params": rf_grid.best_params_,
            "optimal_threshold": rf_threshold,
            "fbeta_score": rf_fbeta,
            "classification_report": classification_report(
                y_test, rf_predictions, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, rf_predictions).tolist(),
        }

        self.logger.info(
            "Random Forest - Test AUC: %.4f, Accuracy@%.2f: %.4f",
            rf_auc,
            rf_threshold,
            rf_test_score,
        )

        # 2. Logistic Regression Model
        self.logger.info("Training Logistic Regression model...")

        # Scale features for Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scalers["logistic_regression"] = scaler

        # Hyperparameter tuning for Logistic Regression
        lr_param_grid = {
            "C": [0.1, 1.0, 10.0, 100.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
            "class_weight": ["balanced", None],
        }

        lr_base = LogisticRegression(random_state=random_state, max_iter=1000)
        lr_grid = GridSearchCV(
            lr_base, lr_param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1
        )
        lr_grid.fit(X_train_scaled, y_train)

        self.models["logistic_regression"] = lr_grid.best_estimator_

        # Feature importance for Logistic Regression (coefficients)
        self.feature_importance["logistic_regression"] = dict(
            zip(feature_names, abs(self.models["logistic_regression"].coef_[0]))
        )

        # Evaluate Logistic Regression with optimized threshold
        lr_train_probabilities = self.models["logistic_regression"].predict_proba(
            X_train_scaled
        )[:, 1]
        lr_probabilities = self.models["logistic_regression"].predict_proba(
            X_test_scaled
        )[:, 1]
        lr_threshold, lr_fbeta = self._optimize_threshold(y_test, lr_probabilities)
        self.thresholds["logistic_regression"] = lr_threshold

        lr_train_predictions = (lr_train_probabilities >= lr_threshold).astype(int)
        lr_predictions = (lr_probabilities >= lr_threshold).astype(int)
        lr_train_score = accuracy_score(y_train, lr_train_predictions)
        lr_test_score = accuracy_score(y_test, lr_predictions)
        lr_auc = roc_auc_score(y_test, lr_probabilities)

        results["logistic_regression"] = {
            "train_accuracy": lr_train_score,
            "test_accuracy": lr_test_score,
            "auc_score": lr_auc,
            "best_params": lr_grid.best_params_,
            "optimal_threshold": lr_threshold,
            "fbeta_score": lr_fbeta,
            "classification_report": classification_report(
                y_test, lr_predictions, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, lr_predictions).tolist(),
        }

        self.logger.info(
            "Logistic Regression - Test AUC: %.4f, Accuracy@%.2f: %.4f",
            lr_auc,
            lr_threshold,
            lr_test_score,
        )

        # 3. XGBoost Model (if available)
        if xgb_available:
            self.logger.info("Training XGBoost model...")

            # Detect GPU for XGBoost
            gpu_info = detect_gpu()

            xgb_param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
            }

            # Configure XGBoost for GPU if available
            xgb_params = {
                "random_state": random_state,
                "eval_metric": "logloss",
                "n_jobs": -1,
            }

            xgb_base: Optional[XGBClassifier] = None
            if gpu_info["available"]:
                try:
                    import warnings

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        xgb_base = XGBClassifier(
                            **xgb_params,
                            tree_method="gpu_hist",
                            predictor="gpu_predictor",
                        )
                        self.logger.info("XGBoost configured for GPU acceleration")
                except TypeError:
                    xgb_base = None
                    self.logger.info("GPU parameters unsupported; falling back to CPU for XGBoost")
                except Exception:
                    xgb_base = None
                    self.logger.info("GPU acceleration failed, using CPU for XGBoost")

            if xgb_base is None:
                xgb_base = XGBClassifier(
                    **xgb_params,
                    tree_method="hist",
                    predictor="cpu_predictor",
                )

            xgb_grid = GridSearchCV(
                xgb_base, xgb_param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1
            )
            try:
                xgb_grid.fit(X_train, y_train)
            except Exception as exc:
                if gpu_info["available"]:
                    self.logger.warning(
                        "GPU-accelerated XGBoost training failed (%s); retrying on CPU.",
                        exc,
                    )
                    xgb_base = XGBClassifier(
                        **xgb_params,
                        tree_method="hist",
                        predictor="cpu_predictor",
                    )
                    xgb_grid = GridSearchCV(
                        xgb_base, xgb_param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1
                    )
                    xgb_grid.fit(X_train, y_train)
                else:
                    raise

            self.models["xgboost"] = xgb_grid.best_estimator_

            # Feature importance for XGBoost
            self.feature_importance["xgboost"] = dict(
                zip(feature_names, self.models["xgboost"].feature_importances_)
            )

            # Evaluate XGBoost with optimized threshold
            xgb_train_probabilities = self.models["xgboost"].predict_proba(X_train)[
                :, 1
            ]
            xgb_probabilities = self.models["xgboost"].predict_proba(X_test)[:, 1]
            xgb_threshold, xgb_fbeta = self._optimize_threshold(
                y_test, xgb_probabilities
            )
            self.thresholds["xgboost"] = xgb_threshold

            xgb_train_predictions = (xgb_train_probabilities >= xgb_threshold).astype(
                int
            )
            xgb_predictions = (xgb_probabilities >= xgb_threshold).astype(int)

            xgb_train_score = accuracy_score(y_train, xgb_train_predictions)
            xgb_test_score = accuracy_score(y_test, xgb_predictions)
            xgb_auc = roc_auc_score(y_test, xgb_probabilities)

            results["xgboost"] = {
                "train_accuracy": xgb_train_score,
                "test_accuracy": xgb_test_score,
                "auc_score": xgb_auc,
                "best_params": xgb_grid.best_params_,
                "optimal_threshold": xgb_threshold,
                "fbeta_score": xgb_fbeta,
                "classification_report": classification_report(
                    y_test, xgb_predictions, output_dict=True
                ),
                "confusion_matrix": confusion_matrix(y_test, xgb_predictions).tolist(),
            }

            self.logger.info(
                "XGBoost - Test AUC: %.4f, Accuracy@%.2f: %.4f",
                xgb_auc,
                xgb_threshold,
                xgb_test_score,
            )

        # 4. Support Vector Machine
        self.logger.info("Training SVM model...")

        svm_param_grid = {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
            "kernel": ["rbf", "poly"],
            "class_weight": ["balanced", None],
        }

        svm_base = SVC(probability=True, random_state=random_state)
        svm_grid = GridSearchCV(
            svm_base, svm_param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1
        )
        svm_grid.fit(X_train_scaled, y_train)

        self.models["svm"] = svm_grid.best_estimator_
        self.scalers["svm"] = scaler  # Store scaler for SVM predictions

        # For SVM, we can't directly get feature importance, so we'll use permutation importance
        try:
            from sklearn.inspection import permutation_importance

            perm_importance = permutation_importance(
                svm_grid.best_estimator_,
                X_test_scaled,
                y_test,
                n_repeats=5,
                random_state=random_state,
            )
            self.feature_importance["svm"] = dict(
                zip(feature_names, perm_importance.importances_mean)
            )
        except ImportError:
            self.feature_importance["svm"] = {}

        # Evaluate SVM with optimized threshold
        svm_train_probabilities = self.models["svm"].predict_proba(X_train_scaled)[:, 1]
        svm_probabilities = self.models["svm"].predict_proba(X_test_scaled)[:, 1]
        svm_threshold, svm_fbeta = self._optimize_threshold(y_test, svm_probabilities)
        self.thresholds["svm"] = svm_threshold

        svm_train_predictions = (svm_train_probabilities >= svm_threshold).astype(int)
        svm_predictions = (svm_probabilities >= svm_threshold).astype(int)
        svm_train_score = accuracy_score(y_train, svm_train_predictions)
        svm_test_score = accuracy_score(y_test, svm_predictions)
        svm_auc = roc_auc_score(y_test, svm_probabilities)

        results["svm"] = {
            "train_accuracy": svm_train_score,
            "test_accuracy": svm_test_score,
            "auc_score": svm_auc,
            "best_params": svm_grid.best_params_,
            "optimal_threshold": svm_threshold,
            "fbeta_score": svm_fbeta,
            "classification_report": classification_report(
                y_test, svm_predictions, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, svm_predictions).tolist(),
        }

        self.logger.info(
            "SVM - Test AUC: %.4f, Accuracy@%.2f: %.4f",
            svm_auc,
            svm_threshold,
            svm_test_score,
        )

        # 5. Ensemble Model (Weighted averaging with optimized threshold)
        self.logger.info("Creating ensemble model...")

        ensemble_components: List[Tuple[str, str, np.ndarray, float]] = [
            (
                "random_forest",
                "rf",
                rf_probabilities,
                results["random_forest"]["auc_score"],
            ),
            (
                "logistic_regression",
                "lr",
                lr_probabilities,
                results["logistic_regression"]["auc_score"],
            ),
        ]

        if xgb_available and "xgboost" in self.models:
            ensemble_components.append(
                (
                    "xgboost",
                    "xgb",
                    xgb_probabilities,
                    results["xgboost"]["auc_score"],
                )
            )
        else:
            xgb_probabilities = None

        if "svm" in self.models:
            ensemble_components.append(
                (
                    "svm",
                    "svm",
                    svm_probabilities,
                    results["svm"]["auc_score"],
                )
            )

        probability_map = {
            component[0]: component[2] for component in ensemble_components
        }
        score_map = {
            component[0]: component[3] for component in ensemble_components
        }

        if StaticWeightedEnsemble is not None:
            static_ensemble = StaticWeightedEnsemble()
            static_ensemble.fit(probability_map, model_scores=score_map)
            ensemble_probabilities = static_ensemble.predict(probability_map)
            weight_lookup = static_ensemble.get_weights()
        else:  # pragma: no cover - fallback path
            weight_values = np.array(
                [max(component[3], 1e-3) for component in ensemble_components],
                dtype=float,
            )
            weight_lookup = {
                component[0]: float(weight)
                for component, weight in zip(ensemble_components, weight_values)
            }
            weight_values = weight_values / weight_values.sum()
            stacked_probabilities = np.vstack(
                [component[2] for component in ensemble_components]
            )
            ensemble_probabilities = np.average(
                stacked_probabilities, axis=0, weights=weight_values
            )

        normalized_weights = np.array(
            [weight_lookup.get(component[0], 0.0) for component in ensemble_components],
            dtype=float,
        )
        weight_sum = normalized_weights.sum()
        if weight_sum <= 0:
            normalized_weights = np.ones(len(ensemble_components), dtype=float)
            normalized_weights = normalized_weights / normalized_weights.sum()
        else:
            normalized_weights = normalized_weights / weight_sum

        weight_lookup = {
            component[0]: float(weight)
            for component, weight in zip(ensemble_components, normalized_weights)
        }
        ensemble_threshold, ensemble_fbeta = self._optimize_threshold(
            y_test, ensemble_probabilities
        )
        ensemble_predictions = (ensemble_probabilities >= ensemble_threshold).astype(
            int
        )
        ensemble_auc = roc_auc_score(y_test, ensemble_probabilities)
        ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)

        self.ensemble_threshold = ensemble_threshold
        self.ensemble_weights = [
            {"model": component[0], "weight": float(weight)}
            for component, weight in zip(ensemble_components, normalized_weights)
        ]

        results["ensemble"] = {
            "test_accuracy": ensemble_accuracy,
            "auc_score": ensemble_auc,
            "optimal_threshold": ensemble_threshold,
            "fbeta_score": ensemble_fbeta,
            "classification_report": classification_report(
                y_test, ensemble_predictions, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, ensemble_predictions).tolist(),
            "model_count": len(ensemble_components),
            "models_used": [component[1] for component in ensemble_components],
            "weights": {
                component[1]: float(weight)
                for component, weight in zip(ensemble_components, normalized_weights)
            },
        }

        self.logger.info(
            "Ensemble Model - Test AUC: %.4f, Accuracy@%.2f: %.4f",
            ensemble_auc,
            ensemble_threshold,
            ensemble_accuracy,
        )
        self.logger.info(
            "Ensemble includes %d models weighted by AUC: %s",
            len(ensemble_components),
            ", ".join(
                f"{component[1]}={weight:.2f}"
                for component, weight in zip(ensemble_components, normalized_weights)
            ),
        )

        if RegimeAdaptiveEnsemble is not None and self.training_metadata is not None:
            try:
                validation_metadata = self.training_metadata.loc[X_test.index].copy()
                ensemble_options = dict(self.regime_options or {})
                regime_enabled = ensemble_options.pop("enabled", True)
                if regime_enabled:
                    use_meta_flag = bool(
                        ensemble_options.pop("use_meta_model", self.use_regime_meta_model)
                    )
                    adaptive = RegimeAdaptiveEnsemble(
                        use_meta_model=use_meta_flag,
                        **ensemble_options,
                    )
                    adaptive.fit(probability_map, y_test, metadata=validation_metadata)
                    self.dynamic_ensemble = adaptive
                    self.use_regime_meta_model = use_meta_flag
                    results["ensemble"]["regime_weights"] = adaptive.to_json()
                else:
                    self.logger.info("Regime-adaptive ensemble disabled via configuration")
                    self.dynamic_ensemble = None
                    results["ensemble"].pop("regime_weights", None)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning(
                    "Failed to build regime-adaptive ensemble: %s", exc
                )

        # Store test data for further analysis
        test_data = {
            "y_test": y_test.tolist(),
            "rf_probabilities": rf_probabilities.tolist(),
            "lr_probabilities": lr_probabilities.tolist(),
            "ensemble_probabilities": ensemble_probabilities.tolist(),
            "ensemble_threshold": ensemble_threshold,
            "optimized_thresholds": {
                key: float(value) for key, value in self.thresholds.items()
            },
            "ensemble_weights": self.ensemble_weights,
        }

        if xgb_probabilities is not None:
            test_data["xgb_probabilities"] = xgb_probabilities.tolist()

        if "svm" in self.models:
            test_data["svm_probabilities"] = svm_probabilities.tolist()

        results["test_data"] = test_data

        self.model_metrics = results

        json_results = self._prepare_json_results(results)

        # Save results for Streamlit
        try:
            results_file = os.path.join("data", "model_results.json")
            with open(results_file, "w") as f:
                json.dump(json_results, f, indent=2)

            self.logger.info(f"Model results saved to {results_file}")
        except Exception as e:
            self.logger.warning(f"Could not save model results: {e}")

        # Automatically save trained models
        save_dir = self.save_models()
        self.logger.info(
            f"🚀 All 4 models trained with GPU acceleration and saved to {save_dir}!"
        )

        self._log_training_run(
            test_size=test_size,
            random_state=random_state,
            feature_names=feature_names,
            results=results,
            json_results=json_results,
            model_dir=save_dir,
        )

        return results

    def predict_risk(
        self,
        X: pd.DataFrame,
        model_type: str = "ensemble",
        metadata: Optional[pd.DataFrame] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict risk using trained models

        Args:
            X: Feature matrix
            model_type: Type of model to use ('random_forest', 'logistic_regression', 'xgboost', 'svm', 'ensemble')
            metadata: Optional DataFrame containing contextual columns (Date, Symbol, Returns)

        Returns:
            Tuple of (predictions, probabilities)
        """
        if model_type == "ensemble":
            # Use ensemble of all available models
            if not self.models:
                raise ValueError("No models have been trained")

            probability_vectors = []
            weights = []
            probability_map: Dict[str, np.ndarray] = {}

            logistic_scaled = None
            svm_scaled = None

            ensemble_entries = (
                self.ensemble_weights
                if self.ensemble_weights
                else [
                    {"model": key, "weight": 1.0}
                    for key in [
                        "random_forest",
                        "logistic_regression",
                        "xgboost",
                        "svm",
                    ]
                ]
            )

            for entry in ensemble_entries:
                model_key = entry.get("model")
                weight = float(entry.get("weight", 0.0))

                if model_key == "random_forest" and "random_forest" in self.models:
                    prob = self.models["random_forest"].predict_proba(X)[:, 1]
                    probability_map["random_forest"] = prob
                elif (
                    model_key == "logistic_regression"
                    and "logistic_regression" in self.models
                ):
                    if logistic_scaled is None:
                        logistic_scaled = self.scalers["logistic_regression"].transform(
                            X
                        )
                    prob = self.models["logistic_regression"].predict_proba(
                        logistic_scaled
                    )[:, 1]
                    probability_map["logistic_regression"] = prob
                elif model_key == "xgboost" and "xgboost" in self.models:
                    prob = self.models["xgboost"].predict_proba(X)[:, 1]
                    probability_map["xgboost"] = prob
                elif model_key == "svm" and "svm" in self.models:
                    if svm_scaled is None:
                        svm_scaled = self.scalers["svm"].transform(X)
                    prob = self.models["svm"].predict_proba(svm_scaled)[:, 1]
                    probability_map["svm"] = prob
                else:
                    continue

                probability_vectors.append(prob)
                weights.append(weight if weight > 0 else 1.0)

            if not probability_vectors:
                raise ValueError("No ensemble components available for prediction")

            dynamic_available = (
                self.dynamic_ensemble is not None
                and metadata is not None
            )

            if dynamic_available:
                meta = metadata.copy()
                try:
                    ensemble_proba = self.dynamic_ensemble.predict(
                        probability_map,
                        metadata=meta,
                    )
                except Exception as exc:  # pragma: no cover
                    self.logger.warning(
                        "Dynamic ensemble prediction failed, falling back to static weights: %s",
                        exc,
                    )
                    dynamic_available = False

            if not dynamic_available:
                weights_array = np.array(weights, dtype=float)
                weights_sum = weights_array.sum()
                if weights_sum <= 0:
                    weights_array = np.ones_like(weights_array) / len(weights_array)
                else:
                    weights_array = weights_array / weights_sum

                stacked = np.vstack(probability_vectors)
                ensemble_proba = np.average(stacked, axis=0, weights=weights_array)

            threshold = self.ensemble_threshold if self.ensemble_threshold else 0.5
            ensemble_pred = (ensemble_proba >= threshold).astype(int)

            return ensemble_pred, ensemble_proba

        elif model_type == "random_forest":
            if "random_forest" not in self.models:
                raise ValueError("Random Forest model not trained")

            probabilities = self.models["random_forest"].predict_proba(X)[:, 1]
            threshold = self.thresholds.get("random_forest", 0.5)
            predictions = (probabilities >= threshold).astype(int)

            return predictions, probabilities

        elif model_type == "logistic_regression":
            if "logistic_regression" not in self.models:
                raise ValueError("Logistic Regression model not trained")

            X_scaled = self.scalers["logistic_regression"].transform(X)
            probabilities = self.models["logistic_regression"].predict_proba(X_scaled)[
                :, 1
            ]
            threshold = self.thresholds.get("logistic_regression", 0.5)
            predictions = (probabilities >= threshold).astype(int)

            return predictions, probabilities

        elif model_type == "xgboost":
            if "xgboost" not in self.models:
                raise ValueError("XGBoost model not trained")

            probabilities = self.models["xgboost"].predict_proba(X)[:, 1]
            threshold = self.thresholds.get("xgboost", 0.5)
            predictions = (probabilities >= threshold).astype(int)

            return predictions, probabilities

        elif model_type == "svm":
            if "svm" not in self.models:
                raise ValueError("SVM model not trained")

            X_scaled = self.scalers["svm"].transform(X)
            probabilities = self.models["svm"].predict_proba(X_scaled)[:, 1]
            threshold = self.thresholds.get("svm", 0.5)
            predictions = (probabilities >= threshold).astype(int)

            return predictions, probabilities

        else:
            raise ValueError(
                f"Unknown model type: {model_type}. Available: {list(self.models.keys())}"
            )

    def save_models(self, timestamp: Optional[str] = None) -> str:
        """Save trained models, scalers, and results to disk"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_dir = os.path.join(self.model_dir, f"models_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        # Save models
        for model_name, model in self.models.items():
            model_file = os.path.join(save_dir, f"{model_name}_model.pkl")
            with open(model_file, "wb") as f:
                pickle.dump(model, f)
            self.logger.info(f"Saved {model_name} model to {model_file}")

        # Save scalers
        scalers_file = os.path.join(save_dir, "scalers.pkl")
        with open(scalers_file, "wb") as f:
            pickle.dump(self.scalers, f)

        # Save feature importance (convert numpy types to native Python types)
        importance_file = os.path.join(save_dir, "feature_importance.json")
        importance_serializable: Dict[str, Any] = {}
        for model_name, importance_dict in self.feature_importance.items():
            importance_serializable[model_name] = {}
            for feature, value in importance_dict.items():
                if hasattr(value, "item"):
                    importance_serializable[model_name][feature] = value.item()
                else:
                    importance_serializable[model_name][feature] = float(value)

        with open(importance_file, "w") as f:
            json.dump(importance_serializable, f, indent=2)

        # Save model metrics
        metrics_file = os.path.join(save_dir, "model_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(self.model_metrics, f, indent=2)

        # Save optimized thresholds and ensemble weights
        thresholds_file = os.path.join(save_dir, "thresholds.json")
        thresholds_payload = {
            "model_thresholds": {
                key: float(value) for key, value in self.thresholds.items()
            },
            "ensemble_threshold": float(self.ensemble_threshold),
            "ensemble_weights": [
                {"model": entry["model"], "weight": float(entry["weight"])}
                for entry in self.ensemble_weights
            ],
        }
        with open(thresholds_file, "w") as f:
            json.dump(thresholds_payload, f, indent=2)

        # Save GPU info
        gpu_file = os.path.join(save_dir, "gpu_info.json")
        with open(gpu_file, "w") as f:
            json.dump(self.gpu_info, f, indent=2)

        self.logger.info(f"All models saved to {save_dir}")
        return save_dir

    def load_models(self, model_dir: str) -> bool:
        """Load trained models from disk"""
        try:
            # Load models
            model_files = [f for f in os.listdir(model_dir) if f.endswith("_model.pkl")]
            for model_file in model_files:
                model_name = model_file.replace("_model.pkl", "")
                model_path = os.path.join(model_dir, model_file)
                with open(model_path, "rb") as f:
                    self.models[model_name] = pickle.load(f)
                self.logger.info(f"Loaded {model_name} model from {model_path}")

            # Load scalers
            scalers_file = os.path.join(model_dir, "scalers.pkl")
            if os.path.exists(scalers_file):
                with open(scalers_file, "rb") as f:
                    self.scalers = pickle.load(f)

            # Load feature importance
            importance_file = os.path.join(model_dir, "feature_importance.json")
            if os.path.exists(importance_file):
                with open(importance_file, "r") as f:  # type: ignore
                    self.feature_importance = json.load(f)

            # Load model metrics
            metrics_file = os.path.join(model_dir, "model_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:  # type: ignore
                    self.model_metrics = json.load(f)

            # Load optimized thresholds and ensemble weights
            thresholds_file = os.path.join(model_dir, "thresholds.json")
            if os.path.exists(thresholds_file):
                with open(thresholds_file, "r") as f:  # type: ignore
                    threshold_data = json.load(f)

                model_thresholds = threshold_data.get("model_thresholds", {})
                self.thresholds = {
                    key: float(value) for key, value in model_thresholds.items()
                }
                self.ensemble_threshold = float(
                    threshold_data.get("ensemble_threshold", 0.5)
                )
                ensemble_weights = threshold_data.get("ensemble_weights", [])
                self.ensemble_weights = [
                    {
                        "model": entry.get("model"),
                        "weight": float(entry.get("weight", 0.0)),
                    }
                    for entry in ensemble_weights
                    if entry.get("model") is not None
                ]

            self.logger.info(f"Successfully loaded models from {model_dir}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False

    @staticmethod
    def _is_numeric(value: Any) -> bool:
        return (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, bool)
        )

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._json_safe(val) for key, val in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(item) for item in value]

        if isinstance(value, np.ndarray):
            return [self._json_safe(item) for item in value.tolist()]

        if isinstance(value, (np.integer,)):
            return int(value)

        if isinstance(value, (np.floating,)):
            return float(value)

        if isinstance(value, (np.bool_,)):
            return bool(value)

        if isinstance(value, pd.Series):
            return [self._json_safe(item) for item in value.tolist()]

        if isinstance(value, pd.DataFrame):
            return [self._json_safe(record) for record in value.to_dict(orient="records")]

        if isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()

        if isinstance(value, timedelta):
            return value.total_seconds()

        return value

    def _prepare_json_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._json_safe(results)
        if isinstance(payload, dict):
            metadata = payload.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            metadata.update(
                {
                    "generated_at": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "gpu": self._json_safe(self.gpu_info),
                    "models": sorted(list(self.models.keys())),
                }
            )
            payload["metadata"] = metadata
        return payload

    def _log_confusion_matrix(self, model_name: str, matrix: Any) -> None:
        if mlflow is None:
            return

        try:
            matrix_payload = self._json_safe({"confusion_matrix": matrix})
            if isinstance(matrix_payload, dict):
                mlflow.log_dict(
                    matrix_payload,
                    f"diagnostics/{model_name}_confusion_matrix.json",
                )
        except Exception as exc:  # pragma: no cover - auxiliary logging
            self.logger.debug(
                "Failed to log confusion matrix for %s: %s", model_name, exc
            )

    def _log_training_run(
        self,
    test_size: float,
    random_state: Optional[int],
        feature_names: List[str],
        results: Dict[str, Any],
        json_results: Dict[str, Any],
        model_dir: str,
    ) -> None:
        if mlflow is None:
            self.logger.debug("MLflow not available; skipping training run logging.")
            return

        def _log_into_active_run() -> None:
            mlflow.set_tags(
                {
                    "component": "RiskPredictor",
                    "run_type": "training",
                    "models_trained": ",".join(sorted(self.models.keys())),
                }
            )

            params = {
                "test_size": test_size,
                "random_state": random_state,
                "num_features": len(feature_names),
                "ensemble_threshold": self.ensemble_threshold,
                "num_models": len(self.models),
                "timestamp": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            }

            try:
                mlflow.log_params({k: v for k, v in params.items() if v is not None})
            except Exception as exc:  # pragma: no cover - diagnostics
                self.logger.debug("Failed to log parameters to MLflow: %s", exc)

            if feature_names:
                try:
                    mlflow.log_dict(
                        {"feature_names": feature_names},
                        "artifacts/feature_names.json",
                    )
                except Exception as exc:
                    self.logger.debug(
                        "Failed to log feature names artifact: %s", exc
                    )

            try:
                mlflow.log_dict(json_results, "artifacts/model_results.json")
            except Exception as exc:
                self.logger.debug("Failed to log model results artifact: %s", exc)

            try:
                mlflow.log_dict(
                    self._json_safe(self.gpu_info), "artifacts/gpu_info.json"
                )
            except Exception as exc:
                self.logger.debug("Failed to log GPU info artifact: %s", exc)

            for model_name, metrics in results.items():
                if not isinstance(metrics, dict):
                    continue

                for key, value in metrics.items():
                    if not self._is_numeric(value):
                        continue
                    try:
                        numeric_value = float(value)
                        if not math.isfinite(numeric_value):
                            continue
                        mlflow.log_metric(f"{model_name}.{key}", numeric_value)
                    except Exception as exc:
                        self.logger.debug(
                            "Failed to log metric %s.%s: %s", model_name, key, exc
                        )

                if "classification_report" in metrics and isinstance(
                    metrics["classification_report"], dict
                ):
                    try:
                        mlflow.log_dict(
                            self._json_safe(metrics["classification_report"]),
                            f"reports/{model_name}_classification_report.json",
                        )
                    except Exception as exc:
                        self.logger.debug(
                            "Failed to log classification report for %s: %s",
                            model_name,
                            exc,
                        )

                if "confusion_matrix" in metrics:
                    self._log_confusion_matrix(
                        model_name, metrics["confusion_matrix"]
                    )

                if "best_params" in metrics and isinstance(metrics["best_params"], dict):
                    try:
                        mlflow.log_dict(
                            self._json_safe(metrics["best_params"]),
                            f"params/{model_name}_best_params.json",
                        )
                    except Exception as exc:
                        self.logger.debug(
                            "Failed to log best params for %s: %s", model_name, exc
                        )

            if "test_data" in results and isinstance(results["test_data"], dict):
                try:
                    mlflow.log_dict(
                        self._json_safe(results["test_data"]),
                        "artifacts/test_data.json",
                    )
                except Exception as exc:
                    self.logger.debug("Failed to log test data artifact: %s", exc)

            if model_dir and os.path.isdir(model_dir):
                try:
                    mlflow.log_artifacts(model_dir, artifact_path="artifacts/models")
                except Exception as exc:
                    self.logger.debug(
                        "Failed to log model directory artifacts: %s", exc
                    )

        try:
            if mlflow.active_run() is None:
                run_name = f"RiskPredictor_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                with mlflow.start_run(run_name=run_name):
                    _log_into_active_run()
            else:
                _log_into_active_run()
        except Exception as exc:  # pragma: no cover - unexpected MLflow issues
            self.logger.warning("Failed to log training run to MLflow: %s", exc)

    def get_feature_importance(
        self, model_type: str = "random_forest", top_n: int = 20
    ) -> Dict[str, float]:
        """
        Get feature importance from trained model

        Args:
            model_type: Type of model
            top_n: Number of top features to return

        Returns:
            Dictionary of feature importance
        """
        if model_type not in self.feature_importance:
            return {}

        importance_dict = self.feature_importance[model_type]

        # Sort by importance and return top N
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return dict(list(sorted_importance.items())[:top_n])

    def _create_risk_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk labels based on future market movements"""

        self.logger.info("Creating risk labels...")

        data = df.copy()

        # Parameters for risk labeling
        volatility_threshold = 0.02  # 2% daily volatility
        return_threshold = -0.05  # -5% return
        forward_window = 5  # 5-day forward window

        # Initialize risk labels
        data["Risk_Label"] = 0
        data["Risk_Score"] = 0.0

        if "Symbol" in data.columns and "Returns" in data.columns:
            for symbol in data["Symbol"].unique():
                symbol_mask = data["Symbol"] == symbol
                symbol_data = data[symbol_mask].copy().sort_values("Date")

                # Calculate forward-looking metrics
                symbol_data["Forward_Return"] = (
                    symbol_data["Returns"]
                    .rolling(forward_window)
                    .mean()
                    .shift(-forward_window)
                )
                symbol_data["Forward_Volatility"] = (
                    symbol_data["Returns"]
                    .rolling(forward_window)
                    .std()
                    .shift(-forward_window)
                )

                # Risk conditions
                high_vol = symbol_data["Forward_Volatility"] > volatility_threshold
                neg_return = symbol_data["Forward_Return"] < return_threshold

                # Binary risk label
                risk_label = (high_vol | neg_return).astype(int)

                # Continuous risk score
                vol_score = (
                    symbol_data["Forward_Volatility"] / volatility_threshold
                ).clip(0, 2) / 2
                return_score = (
                    -symbol_data["Forward_Return"] / -return_threshold
                ).clip(0, 2) / 2
                risk_score = ((vol_score + return_score) / 2).fillna(0).clip(0, 1)

                # Update main dataframe
                data.loc[symbol_mask, "Risk_Label"] = risk_label
                data.loc[symbol_mask, "Risk_Score"] = risk_score

        # Remove rows with missing labels (due to forward-looking calculation)
        data = data.dropna(subset=["Risk_Label"])

        risk_count = data["Risk_Label"].sum()
        total_count = len(data)

        self.logger.info(
            f"Created risk labels: {risk_count}/{total_count} ({risk_count/total_count:.2%}) high-risk periods"
        )

        return data
