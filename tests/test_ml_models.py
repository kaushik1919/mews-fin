"""
Unit tests for ML models module.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml_models import MLModelTrainer, ModelEnsemble


class TestMLModelTrainer:
    """Test cases for MLModelTrainer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        data = {
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'risk_label': np.random.choice([0, 1], 100)
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def trainer(self):
        """Create MLModelTrainer instance."""
        return MLModelTrainer()
    
    def test_trainer_initialization(self, trainer):
        """Test trainer initialization."""
        assert trainer is not None
        assert hasattr(trainer, 'models')
        assert isinstance(trainer.models, dict)
    
    def test_prepare_features(self, trainer, sample_data):
        """Test feature preparation."""
        X, y = trainer.prepare_features(sample_data, target_column='risk_label')
        
        assert X.shape[0] == 100
        assert X.shape[1] == 3  # 3 features
        assert len(y) == 100
        assert set(y.unique()) <= {0, 1}
    
    def test_train_models(self, trainer, sample_data):
        """Test model training."""
        X, y = trainer.prepare_features(sample_data, target_column='risk_label')
        
        # Mock GPU detection to avoid CUDA dependencies in tests
        with patch('ml_models.torch.cuda.is_available', return_value=False):
            results = trainer.train_models(X, y)
        
        assert isinstance(results, dict)
        assert 'Random Forest' in results
        assert 'XGBoost' in results
        assert all('auc' in result for result in results.values())
    
    def test_model_prediction(self, trainer, sample_data):
        """Test model predictions."""
        X, y = trainer.prepare_features(sample_data, target_column='risk_label')
        
        with patch('ml_models.torch.cuda.is_available', return_value=False):
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
            'model1': np.array([0.1, 0.3, 0.8, 0.6]),
            'model2': np.array([0.2, 0.4, 0.7, 0.5]),
            'model3': np.array([0.15, 0.35, 0.75, 0.55])
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


@pytest.mark.integration
class TestMLModelIntegration:
    """Integration tests for ML models."""
    
    def test_full_pipeline(self):
        """Test complete ML pipeline."""
        # Create larger sample dataset
        np.random.seed(42)
        n_samples = 1000
        data = {
            'price_change': np.random.randn(n_samples),
            'volume_change': np.random.randn(n_samples),
            'volatility': np.random.exponential(0.5, n_samples),
            'rsi': np.random.uniform(0, 100, n_samples),
            'macd': np.random.randn(n_samples),
        }
        
        # Create realistic risk labels based on features
        risk_score = (
            data['price_change'] * -0.3 +
            data['volatility'] * 0.5 +
            np.random.randn(n_samples) * 0.1
        )
        data['risk_label'] = (risk_score > np.percentile(risk_score, 70)).astype(int)
        
        df = pd.DataFrame(data)
        
        trainer = MLModelTrainer()
        X, y = trainer.prepare_features(df, target_column='risk_label')
        
        with patch('ml_models.torch.cuda.is_available', return_value=False):
            results = trainer.train_models(X, y)
            predictions = trainer.predict(X)
        
        # Check that we get reasonable performance
        assert all(result['auc'] > 0.5 for result in results.values())
        assert len(predictions) == len(X)
        
        # Test ensemble
        ensemble = ModelEnsemble()
        model_predictions = {name: trainer.models[name].predict_proba(X)[:, 1] 
                           for name in trainer.models.keys()}
        ensemble_pred = ensemble.average_predictions(model_predictions)
        
        assert len(ensemble_pred) == len(X)
        assert all(0 <= pred <= 1 for pred in ensemble_pred)
