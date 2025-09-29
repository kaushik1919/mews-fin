"""
GPU-Accelerated ML Model Training Script
Trains all 4 models (Random Forest, Logistic Regression, XGBoost, SVM) with GPU support
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import sys
sys.path.append('.')

from src.ml_models import RiskPredictor, detect_gpu
from src.data_integrator import DataIntegrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("=" * 60)
    print("🚀 GPU-Accelerated ML Model Training")
    print("=" * 60)
    
    # Check GPU
    gpu_info = detect_gpu()
    if gpu_info['available']:
        print(f"✅ GPU Detected: {gpu_info['name']}")
        print(f"🔥 CUDA Device: {gpu_info['device']}")
    else:
        print("⚠️  No GPU detected, using CPU")
    
    print("\n" + "=" * 60)
    print("📊 Loading Data")
    print("=" * 60)
    
    # Load integrated dataset
    data_file = "data/integrated_dataset.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("Please run: python main.py --full-pipeline first")
        return
    
    df = pd.read_csv(data_file)
    print(f"✅ Loaded dataset: {len(df)} samples, {len(df.columns)} features")
    
    # Initialize predictor
    predictor = RiskPredictor()
    
    print("\n" + "=" * 60)
    print("🤖 Training All 4 Models with GPU Acceleration")
    print("=" * 60)
    
    # Prepare data for modeling
    X, y, feature_names = predictor.prepare_modeling_data(df)
    print(f"✅ Prepared features: {len(feature_names)} features")
    print(f"✅ Target distribution: {np.bincount(y)}")
    
    # Train all models
    start_time = datetime.now()
    print(f"\n🔥 Starting training at {start_time}")
    
    results = predictor.train_models(
        X, y, 
        feature_names=feature_names,
        test_size=0.2,
        random_state=42
    )
    
    end_time = datetime.now()
    training_duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("📈 Training Results Summary")
    print("=" * 60)
    
    print(f"⏱️  Total Training Time: {training_duration}")
    print(f"💾 Models Saved: {len(predictor.models)} models")
    
    # Display results
    model_count = 0
    for model_name, metrics in results.items():
        if model_name not in ['test_data'] and isinstance(metrics, dict):
            model_count += 1
            accuracy = metrics.get('test_accuracy', 0)
            auc = metrics.get('auc_score', 0)
            print(f"  {model_count}. {model_name.replace('_', ' ').title():<20} "
                  f"Accuracy: {accuracy:.4f}  AUC: {auc:.4f}")
    
    # Ensemble results
    if 'ensemble' in results:
        ensemble_acc = results['ensemble'].get('test_accuracy', 0)
        ensemble_auc = results['ensemble'].get('auc_score', 0)
        model_names = results['ensemble'].get('models_used', [])
        print(f"\n🎯 Ensemble Model ({len(model_names)} models): "
              f"Accuracy: {ensemble_acc:.4f}  AUC: {ensemble_auc:.4f}")
        print(f"   Models used: {', '.join(model_names)}")
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    
    # Quick load test
    print("\n🔄 Testing Model Loading...")
    test_predictor = RiskPredictor()
    
    # Find latest model directory
    model_dirs = [d for d in os.listdir('models') if d.startswith('models_')]
    if model_dirs:
        latest_dir = os.path.join('models', sorted(model_dirs)[-1])
        if test_predictor.load_models(latest_dir):
            print(f"✅ Successfully loaded models from {latest_dir}")
            print(f"📁 Available models: {list(test_predictor.models.keys())}")
            
            # Test prediction
            if len(X) > 0:
                sample_X = X[:5]  # Test with 5 samples
                for model_type in test_predictor.models.keys():
                    try:
                        predictions, probabilities = test_predictor.predict_risk(sample_X, model_type)
                        print(f"✅ {model_type} prediction test passed")
                    except Exception as e:
                        print(f"❌ {model_type} prediction test failed: {e}")
        else:
            print("❌ Failed to load models")
    
    print("\n🎉 All done! Your GPU-trained models are ready to use!")
    print("💡 You can now run the Streamlit app: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()
