"""
Unit tests for ML models module.
"""

import numpy as np
import pandas as pd
import pytest

from src.ensemble.regime import RegimeAdaptiveEnsemble, VolatilityRegimeDetector
from src.ml_models import MLModelTrainer, ModelEnsemble, RiskPredictor


class TestMLModelTrainer:
    """Test cases for MLModelTrainer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        data = {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
            "risk_label": np.random.choice([0, 1], 100),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def trainer(self):
        """Create MLModelTrainer instance."""
        return MLModelTrainer()

    def test_trainer_initialization(self, trainer):
        """Test trainer initialization."""
        assert trainer is not None
        assert hasattr(trainer, "models")
        assert isinstance(trainer.models, dict)

    def test_prepare_features(self, trainer, sample_data):
        """Test feature preparation."""
        X, y = trainer.prepare_features(sample_data, target_column="risk_label")

        assert X.shape[0] == 100
        assert X.shape[1] == 3  # 3 features
        assert len(y) == 100
        assert set(y.unique()) <= {0, 1}

    def test_train_models(self, trainer, sample_data):
        """Test model training."""
        X, y = trainer.prepare_features(sample_data, target_column="risk_label")

        # Mock GPU detection to avoid CUDA dependencies in tests
        results = trainer.train_models(X, y)

        assert isinstance(results, dict)
        assert "Random Forest" in results
        assert "XGBoost" in results
        assert all("auc" in result for result in results.values())

    def test_model_prediction(self, trainer, sample_data):
        """Test model predictions."""
        X, y = trainer.prepare_features(sample_data, target_column="risk_label")

        trainer.train_models(X, y)
        predictions = trainer.predict(X)

        assert len(predictions) == len(X)
        assert all(0 <= pred <= 1 for pred in predictions)


class TestModelEnsemble:
    """Test cases for ModelEnsemble class."""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions from multiple models."""
        return {
            "model1": np.array([0.1, 0.3, 0.8, 0.6]),
            "model2": np.array([0.2, 0.4, 0.7, 0.5]),
            "model3": np.array([0.15, 0.35, 0.75, 0.55]),
        }

    def test_ensemble_averaging(self, sample_predictions):
        """Test ensemble averaging."""
        ensemble = ModelEnsemble()
        avg_predictions = ensemble.average_predictions(sample_predictions)

        expected = np.array([0.15, 0.35, 0.75, 0.55])
        np.testing.assert_array_almost_equal(avg_predictions, expected)

    def test_ensemble_prediction_bounds(self, sample_predictions):
        """Test that ensemble predictions are within valid bounds."""
        ensemble = ModelEnsemble()
        predictions = ensemble.average_predictions(sample_predictions)

        assert all(0 <= pred <= 1 for pred in predictions)


class TestRiskPredictorUtilities:
    """Tests for RiskPredictor helper utilities."""

    def test_prepare_json_results_serializes_numpy(self):
        predictor = RiskPredictor()
        predictor.models = {"random_forest": object()}
        predictor.gpu_info = {"available": True, "device": "cpu", "name": "Test"}

        results = {
            "random_forest": {
                "train_accuracy": np.float32(0.95),
                "test_accuracy": np.float64(0.91),
                "auc_score": np.float64(0.88),
                "confusion_matrix": np.array([[12, 3], [4, 9]]),
                "classification_report": {
                    "0": {"precision": np.float32(0.7), "recall": np.float64(0.6)},
                    "1": {"precision": np.float32(0.8), "recall": np.float64(0.75)},
                },
            },
            "test_data": {
                "y_test": np.array([0, 1, 0], dtype=np.int64),
                "ensemble_threshold": np.float32(0.55),
            },
        }

        json_results = predictor._prepare_json_results(results)

        assert isinstance(json_results["random_forest"]["train_accuracy"], float)
        assert json_results["random_forest"]["confusion_matrix"] == [[12, 3], [4, 9]]
        assert "metadata" in json_results
        assert json_results["metadata"]["models"] == ["random_forest"]
        assert json_results["metadata"]["gpu"]["name"] == "Test"
        assert json_results["test_data"]["y_test"] == [0, 1, 0]

    def test_log_training_run_handles_missing_mlflow(self, tmp_path, monkeypatch):
        predictor = RiskPredictor()
        predictor.models = {"random_forest": object()}

        results = {"random_forest": {"train_accuracy": 0.9}}
        json_results = predictor._prepare_json_results(results)

        model_dir = tmp_path / "model_dir"
        model_dir.mkdir()

        monkeypatch.setattr("src.ml_models.mlflow", None)

        assert (
            predictor._log_training_run(
                test_size=0.2,
                random_state=42,
                feature_names=["feature_a", "feature_b"],
                results=results,
                json_results=json_results,
                model_dir=str(model_dir),
            )
            is None
        )

    def test_regime_adaptive_ensemble_serialization(self):
        rng = np.random.default_rng(0)
        probabilities = {
            "random_forest": rng.random(120),
            "logistic_regression": rng.random(120),
            "xgboost": rng.random(120),
        }
        targets = (rng.random(120) > 0.5).astype(int)
        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        metadata = pd.DataFrame(
            {
                "Date": dates,
                "Symbol": ["TEST"] * 120,
                "Returns": rng.normal(0, 0.02, size=120),
            }
        )

        ensemble = RegimeAdaptiveEnsemble()
        ensemble.fit(probabilities, targets, metadata=metadata)
        summary = ensemble.to_json()

        assert "default" in summary
        assert "regimes" in summary
        assert summary["meta_model"]["enabled"] in {True, False}

        preds = ensemble.predict(probabilities, metadata=metadata)
        assert preds.shape == (120,)
        assert np.all((preds >= 0) & (preds <= 1))


class TestVolatilityRegimeDetector:
    def test_detector_assigns_regimes(self):
        rng = np.random.default_rng(1)
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        df = pd.DataFrame(
            {
                "Date": dates,
                "Symbol": ["SYM"] * 60,
                "Returns": rng.normal(0, 0.02, size=60),
            }
        )

        detector = VolatilityRegimeDetector(lookback=10)
        regimes = detector.fit_transform(df)

        assert len(regimes) == len(df)
        assert set(regimes.unique()) <= {
            "low_volatility",
            "moderate_volatility",
            "high_volatility",
        }


@pytest.mark.integration
class TestMLModelIntegration:
    """Integration tests for ML models."""

    def test_full_pipeline(self):
        """Test complete ML pipeline."""
        # Create larger sample dataset
        np.random.seed(42)
        n_samples = 1000
        data = {
            "price_change": np.random.randn(n_samples),
            "volume_change": np.random.randn(n_samples),
            "volatility": np.random.exponential(0.5, n_samples),
            "rsi": np.random.uniform(0, 100, n_samples),
            "macd": np.random.randn(n_samples),
        }

        # Create realistic risk labels based on features
        risk_score = (
            data["price_change"] * -0.3
            + data["volatility"] * 0.5
            + np.random.randn(n_samples) * 0.1
        )
        data["risk_label"] = (risk_score > np.percentile(risk_score, 70)).astype(int)

        df = pd.DataFrame(data)

        trainer = MLModelTrainer()
        X, y = trainer.prepare_features(df, target_column="risk_label")

        results = trainer.train_models(X, y)
        predictions = trainer.predict(X)

        # Check that we get reasonable performance
        assert all(result["auc"] >= 0.5 for result in results.values())
        assert len(predictions) == len(X)

        # Test ensemble
        ensemble = ModelEnsemble()
        model_predictions = {
            name: trainer.models[name].predict_proba(X)[:, 1]
            for name in trainer.models.keys()
        }
        ensemble_pred = ensemble.average_predictions(model_predictions)

        assert len(ensemble_pred) == len(X)
        assert all(0 <= pred <= 1 for pred in ensemble_pred)
