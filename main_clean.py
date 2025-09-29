#!/usr/bin/env python3
"""
Market Risk Early Warning System (MEWS)
A comprehensive ML system for predicting market risk using free data sources.

Usage:
    python main.py --help
    python main.py --symbols AAPL MSFT --start-date 2023-01-01
    streamlit run streamlit_app.py
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from data_fetcher import DataFetcher
from data_preprocessor import DataPreprocessor
from ml_models import RiskPredictor
from visualizer import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/mews.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ['data', 'models', 'outputs', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Market Risk Early Warning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --symbols AAPL MSFT GOOGL
  python main.py --full-pipeline --start-date 2020-01-01
  python main.py --train-only --symbols AAPL
  streamlit run streamlit_app.py
        """
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
        help='Stock symbols to analyze (default: AAPL MSFT GOOGL TSLA NVDA)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='2020-01-01',
        help='Start date for data collection (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date for data collection (YYYY-MM-DD, default: today)'
    )
    
    parser.add_argument(
        '--full-pipeline',
        action='store_true',
        help='Run complete pipeline: fetch data, train models, generate reports'
    )
    
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Only train models on existing data'
    )
    
    parser.add_argument(
        '--predict-only',
        action='store_true',
        help='Only generate predictions using existing models'
    )
    
    parser.add_argument(
        '--visualize-only',
        action='store_true',
        help='Only generate visualizations using existing data'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='.env',
        help='Configuration file path (default: .env)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize components
        logger.info("Initializing Market Risk Early Warning System...")
        create_directories()
        
        config = Config(config_file=args.config)
        data_fetcher = DataFetcher(config)
        preprocessor = DataPreprocessor(config)
        risk_predictor = RiskPredictor(config)
        visualizer = Visualizer(config)
        
        # Execute based on arguments
        if args.full_pipeline:
            run_full_pipeline(
                data_fetcher, preprocessor, risk_predictor, visualizer,
                args.symbols, args.start_date, args.end_date
            )
        elif args.train_only:
            run_training_only(risk_predictor)
        elif args.predict_only:
            run_prediction_only(risk_predictor, args.symbols)
        elif args.visualize_only:
            run_visualization_only(visualizer, args.symbols)
        else:
            # Default: run full pipeline
            logger.info("No specific mode selected, running full pipeline...")
            run_full_pipeline(
                data_fetcher, preprocessor, risk_predictor, visualizer,
                args.symbols, args.start_date, args.end_date
            )
        
        logger.info("MEWS execution completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        sys.exit(1)


def run_full_pipeline(
    data_fetcher: DataFetcher,
    preprocessor: DataPreprocessor,
    risk_predictor: RiskPredictor,
    visualizer: Visualizer,
    symbols: List[str],
    start_date: str,
    end_date: Optional[str]
):
    """Run the complete MEWS pipeline."""
    logger.info("Running full MEWS pipeline...")
    
    # Step 1: Fetch data
    logger.info("Step 1: Fetching market data...")
    stock_data = data_fetcher.fetch_stock_data(symbols, start_date, end_date)
    
    # Step 2: Preprocess data
    logger.info("Step 2: Preprocessing data...")
    processed_data = preprocessor.process_data(stock_data)
    
    # Step 3: Train models
    logger.info("Step 3: Training ML models...")
    training_results = risk_predictor.train_and_evaluate(processed_data)
    
    # Step 4: Generate predictions
    logger.info("Step 4: Generating predictions...")
    predictions = risk_predictor.predict(processed_data)
    
    # Step 5: Create visualizations
    logger.info("Step 5: Creating visualizations...")
    visualizer.create_comprehensive_report(
        processed_data, predictions, training_results
    )
    
    logger.info("Full pipeline completed successfully!")


def run_training_only(risk_predictor: RiskPredictor):
    """Run only model training on existing data."""
    logger.info("Running training-only mode...")
    
    # Load existing processed data
    data_file = Path("data/processed_data.csv")
    if not data_file.exists():
        raise FileNotFoundError(
            "No processed data found. Run with --full-pipeline first."
        )
    
    import pandas as pd
    processed_data = pd.read_csv(data_file)
    
    # Train models
    training_results = risk_predictor.train_and_evaluate(processed_data)
    logger.info("Training completed successfully!")


def run_prediction_only(risk_predictor: RiskPredictor, symbols: List[str]):
    """Run only prediction using existing models."""
    logger.info("Running prediction-only mode...")
    
    # Load models and generate predictions
    predictions = risk_predictor.load_and_predict(symbols)
    logger.info("Predictions generated successfully!")


def run_visualization_only(visualizer: Visualizer, symbols: List[str]):
    """Run only visualization generation."""
    logger.info("Running visualization-only mode...")
    
    # Load existing data and create visualizations
    data_file = Path("data/processed_data.csv")
    if not data_file.exists():
        raise FileNotFoundError(
            "No processed data found. Run with --full-pipeline first."
        )
    
    import pandas as pd
    processed_data = pd.read_csv(data_file)
    
    visualizer.create_comprehensive_report(processed_data, None, None)
    logger.info("Visualizations created successfully!")


if __name__ == "__main__":
    main()
