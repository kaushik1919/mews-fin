"""
Quick test script to create sample model results for Streamlit demo
"""
import json
import pandas as pd
import numpy as np

# Create sample model results similar to what the real system would produce
sample_results = {
    "random_forest": {
        "train_accuracy": 0.9012,
        "test_accuracy": 0.8385,
        "auc_score": 0.9223,
        "best_params": {
            "n_estimators": 300,
            "max_depth": 20,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
            "max_features": "sqrt"
        },
        "classification_report": {
            "0": {"precision": 0.87, "recall": 0.91, "f1-score": 0.89, "support": 316},
            "1": {"precision": 0.78, "recall": 0.69, "f1-score": 0.73, "support": 136},
            "accuracy": 0.84,
            "macro avg": {"precision": 0.82, "recall": 0.80, "f1-score": 0.81, "support": 452},
            "weighted avg": {"precision": 0.84, "recall": 0.84, "f1-score": 0.84, "support": 452}
        },
        "confusion_matrix": [[288, 28], [42, 94]],
        "feature_importance": {
            "volatility": 0.142,
            "rsi": 0.098,
            "volume_sma": 0.087,
            "bollinger_position": 0.076,
            "price_change_pct": 0.065,
            "macd_signal": 0.058,
            "sentiment_finbert": 0.051,
            "volume_ratio": 0.047,
            "momentum": 0.043,
            "price_trend": 0.039
        }
    },
    "logistic_regression": {
        "train_accuracy": 0.8156,
        "test_accuracy": 0.7854,
        "auc_score": 0.8241,
        "best_params": {
            "C": 1.0,
            "penalty": "l2",
            "solver": "liblinear",
            "class_weight": "balanced",
            "max_iter": 1000
        },
        "classification_report": {
            "0": {"precision": 0.83, "recall": 0.88, "f1-score": 0.85, "support": 316},
            "1": {"precision": 0.68, "recall": 0.58, "f1-score": 0.63, "support": 136},
            "accuracy": 0.79,
            "macro avg": {"precision": 0.75, "recall": 0.73, "f1-score": 0.74, "support": 452},
            "weighted avg": {"precision": 0.78, "recall": 0.79, "f1-score": 0.78, "support": 452}
        },
        "confusion_matrix": [[278, 38], [57, 79]]
    },
    "xgboost": {
        "train_accuracy": 0.9234,
        "test_accuracy": 0.8540,
        "auc_score": 0.9156,
        "best_params": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "device": "cuda"
        },
        "classification_report": {
            "0": {"precision": 0.88, "recall": 0.92, "f1-score": 0.90, "support": 316},
            "1": {"precision": 0.81, "recall": 0.71, "f1-score": 0.76, "support": 136},
            "accuracy": 0.85,
            "macro avg": {"precision": 0.84, "recall": 0.82, "f1-score": 0.83, "support": 452},
            "weighted avg": {"precision": 0.86, "recall": 0.85, "f1-score": 0.85, "support": 452}
        },
        "confusion_matrix": [[291, 25], [40, 96]]
    },
    "svm": {
        "train_accuracy": 0.8421,
        "test_accuracy": 0.7965,
        "auc_score": 0.8387,
        "best_params": {
            "C": 10.0,
            "gamma": "scale",
            "kernel": "rbf",
            "class_weight": "balanced"
        },
        "classification_report": {
            "0": {"precision": 0.84, "recall": 0.89, "f1-score": 0.86, "support": 316},
            "1": {"precision": 0.71, "recall": 0.61, "f1-score": 0.66, "support": 136},
            "accuracy": 0.80,
            "macro avg": {"precision": 0.77, "recall": 0.75, "f1-score": 0.76, "support": 452},
            "weighted avg": {"precision": 0.79, "recall": 0.80, "f1-score": 0.79, "support": 452}
        },
        "confusion_matrix": [[282, 34], [53, 83]]
    },
    "ensemble": {
        "test_accuracy": 0.8697,
        "auc_score": 0.9287,
        "model_count": 4,
        "models_used": ["rf", "lr", "xgb", "svm"],
        "classification_report": {
            "0": {"precision": 0.89, "recall": 0.93, "f1-score": 0.91, "support": 316},
            "1": {"precision": 0.83, "recall": 0.74, "f1-score": 0.78, "support": 136},
            "accuracy": 0.87,
            "macro avg": {"precision": 0.86, "recall": 0.84, "f1-score": 0.85, "support": 452},
            "weighted avg": {"precision": 0.87, "recall": 0.87, "f1-score": 0.87, "support": 452}
        },
        "confusion_matrix": [[294, 22], [35, 101]]
    },
    "test_data": {
        "y_test": [0] * 316 + [1] * 136,
        "rf_probabilities": np.random.beta(2, 5, 452).tolist(),
        "lr_probabilities": np.random.beta(2, 6, 452).tolist(),
        "xgb_probabilities": np.random.beta(2, 4.5, 452).tolist(),
        "svm_probabilities": np.random.beta(2, 5.5, 452).tolist(),
        "ensemble_probabilities": np.random.beta(2, 4, 452).tolist()
    }
}

# Save to JSON file for Streamlit
with open('data/model_results.json', 'w') as f:
    json.dump(sample_results, f, indent=2)

print("Sample model results created for Streamlit demo!")
print("Results include 4 models: Random Forest, Logistic Regression, XGBoost, SVM + Ensemble")
print("Best performing model: Ensemble (92.87% AUC, 86.97% Accuracy)")
print("GPU-accelerated XGBoost: 91.56% AUC, 85.40% Accuracy")
