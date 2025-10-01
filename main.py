"""
Main orchestrator for the Market Risk Early Warning System
Coordinates all components and provides CLI interface
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    from src.backtester import RiskBacktester
    from src.config import Config
    from src.data_fetcher import StockDataFetcher
    from src.data_integrator import DataIntegrator
    from src.data_preprocessor import DataPreprocessor
    from src.ml_models import RiskPredictor
    from src.news_collector import NewsDataCollector
    from src.sec_downloader import SECFilingsDownloader
    from src.sentiment_analyzer import SentimentAnalyzer
    from src.visualizer import RiskVisualizer
    from src.utils.logging import get_logger
    from src.research import (
        GraphFeatureAblation,
        ResearchEvaluator,
        ResearchReportBuilder,
        RobustnessStressTester,
        SentimentBiasDetector,
        SentimentImpactTester,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print(
        "Please ensure all required packages are installed: pip install -r requirements.txt"
    )
    sys.exit(1)


class MarketRiskSystem:
    """Main orchestrator for the Market Risk Early Warning System"""

    def __init__(self, config_path: str = None):
        """Initialize the system with configuration"""

        # Setup configuration
        self.config = Config()
        if config_path and os.path.exists(config_path):
            # Load custom config if provided
            pass

        # Setup logging
        self.config.setup_logging()
        self.config.setup_mlflow()
        self.logger = get_logger(__name__)

        # Validate configuration
        config_valid = self.config.validate_config()
        if not config_valid:
            self.logger.warning(
                "Some API keys are missing - functionality will be limited"
            )

        # Initialize components
        self.stock_fetcher = StockDataFetcher(self.config.ALPHA_VANTAGE_API_KEY)
        self.sec_downloader = SECFilingsDownloader()
        self.news_collector = NewsDataCollector(
            self.config.GNEWS_API_KEY, self.config.NEWS_API_KEY
        )
        self.preprocessor = DataPreprocessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.data_integrator = DataIntegrator()
        self.ml_models = RiskPredictor()
        self.visualizer = RiskVisualizer(self.config.OUTPUT_DIR)
        self.backtester = RiskBacktester()

        # Data storage
        self.stock_data = None
        self.news_data = None
        self.sec_data = None
        self.integrated_data = None
        self.predictions = None

        self.logger.info("Market Risk Early Warning System initialized")

    def run_full_pipeline(
        self,
        symbols: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        skip_data_collection: bool = False,
    ) -> Dict:
        """
        Run the complete pipeline from data collection to backtesting

        Args:
            symbols: List of stock symbols (uses config default if None)
            start_date: Start date for data collection
            end_date: End date for data collection
            skip_data_collection: Skip data collection and use existing data

        Returns:
            Dictionary with pipeline results
        """
        self.logger.info("Starting full pipeline execution...")

        # Set defaults
        if symbols is None:
            symbols = self.config.SP500_SYMBOLS[:20]  # Use first 20 for demo

        if start_date is None:
            start_date = self.config.START_DATE

        if end_date is None:
            end_date = self.config.END_DATE

        results = {
            "pipeline_start": datetime.now().isoformat(),
            "symbols_processed": symbols,
            "date_range": {"start": start_date, "end": end_date},
            "stages_completed": [],
            "errors": [],
        }

        try:
            # Stage 1: Data Collection
            if not skip_data_collection:
                self.logger.info("=== Stage 1: Data Collection ===")
                self.collect_all_data(symbols, start_date, end_date)
                results["stages_completed"].append("data_collection")
            else:
                self.logger.info("Skipping data collection - loading existing data")
                self.load_existing_data()

            # Stage 2: Data Preprocessing
            self.logger.info("=== Stage 2: Data Preprocessing ===")
            self.preprocess_all_data()
            results["stages_completed"].append("data_preprocessing")

            # Stage 3: Sentiment Analysis
            self.logger.info("=== Stage 3: Sentiment Analysis ===")
            self.analyze_sentiment()
            results["stages_completed"].append("sentiment_analysis")

            # Stage 4: Data Integration
            self.logger.info("=== Stage 4: Data Integration ===")
            feature_groups = self.integrate_all_data()
            results["stages_completed"].append("data_integration")
            results["feature_groups"] = {k: len(v) for k, v in feature_groups.items()}

            # Stage 5: Model Training
            self.logger.info("=== Stage 5: Model Training ===")
            model_results = self.train_models()
            results["stages_completed"].append("model_training")
            results["model_performance"] = model_results

            # Stage 6: Predictions
            self.logger.info("=== Stage 6: Generating Predictions ===")
            self.generate_predictions()
            results["stages_completed"].append("predictions")

            # Stage 7: Visualizations
            self.logger.info("=== Stage 7: Creating Visualizations ===")
            viz_paths = self.create_visualizations(symbols[:5])  # Top 5 symbols for viz
            results["stages_completed"].append("visualizations")
            results["visualization_paths"] = viz_paths

            # Stage 8: Backtesting
            self.logger.info("=== Stage 8: Backtesting ===")
            backtest_results = self.run_backtesting()
            results["stages_completed"].append("backtesting")
            results["backtest_summary"] = backtest_results

            # Stage 9: Final Report
            self.logger.info("=== Stage 9: Generating Final Report ===")
            report_path = self.generate_final_report(results)
            results["stages_completed"].append("final_report")
            results["final_report_path"] = report_path

            self.logger.info("=== Stage 10: Research Addendum ===")
            research_payload = self.generate_research_artifacts()
            if research_payload:
                results["stages_completed"].append("research_addendum")
                results["research_artifacts"] = research_payload

            results["pipeline_end"] = datetime.now().isoformat()
            results["pipeline_status"] = "completed"

            self.logger.info("Full pipeline completed successfully!")

        except Exception as e:
            error_msg = f"Pipeline failed at stage {len(results['stages_completed']) + 1}: {str(e)}"
            self.logger.error(error_msg)
            results["errors"].append(error_msg)
            results["pipeline_status"] = "failed"
            results["pipeline_end"] = datetime.now().isoformat()

        return results

    def collect_all_data(self, symbols: List[str], start_date: str, end_date: str):
        """Collect data from all sources"""

        # 1. Stock Data
        self.logger.info("Collecting stock market data...")
        self.stock_data = self.stock_fetcher.fetch_all_data(
            symbols,
            start_date,
            end_date,
            self.config.DATA_DIR,
            use_alpha_vantage=bool(self.config.ALPHA_VANTAGE_API_KEY),
        )

        # 2. News Data
        if self.config.GNEWS_API_KEY or self.config.NEWS_API_KEY:
            self.logger.info("Collecting news data...")
            self.news_data = self.news_collector.fetch_news_for_symbols(
                symbols,
                start_date,
                end_date,
                max_articles_per_symbol=20,  # Limit for free tiers
                use_all_sources=True,
            )
            self.news_collector.save_news_data(self.news_data, self.config.DATA_DIR)
        else:
            self.logger.warning("No news API keys provided - skipping news collection")
            self.news_data = None

        # 3. SEC Filings Data
        self.logger.info("Collecting SEC filings data...")
        self.sec_data = self.sec_downloader.fetch_filings_for_symbols(
            symbols[:10], max_filings_per_symbol=3  # Limit to avoid rate limits
        )
        if not self.sec_data.empty:
            self.sec_downloader.save_filings_data(self.sec_data, self.config.DATA_DIR)

    def load_existing_data(self):
        """Load existing data from files"""
        try:
            import pandas as pd

            # Load stock data
            stock_file = os.path.join(self.config.DATA_DIR, "combined_stock_data.csv")
            if os.path.exists(stock_file):
                self.stock_data = {"combined": pd.read_csv(stock_file)}
                self.logger.info("Loaded existing stock data")

            # Load news data
            news_file = os.path.join(self.config.DATA_DIR, "news_data.csv")
            if os.path.exists(news_file):
                self.news_data = pd.read_csv(news_file)
                self.logger.info("Loaded existing news data")

            # Load SEC data
            sec_file = os.path.join(self.config.DATA_DIR, "sec_filings.csv")
            if os.path.exists(sec_file):
                self.sec_data = pd.read_csv(sec_file)
                self.logger.info("Loaded existing SEC data")

        except Exception as e:
            self.logger.error(f"Error loading existing data: {str(e)}")

    def preprocess_all_data(self):
        """Preprocess all collected data"""

        # Process stock data
        if self.stock_data:
            if isinstance(self.stock_data, dict):
                # If it's a dictionary, combine all DataFrames
                stock_dfs = []
                for symbol, df in self.stock_data.items():
                    if not df.empty:
                        # Ensure proper structure
                        df_copy = df.copy()

                        # Ensure index is clean
                        df_copy = df_copy.reset_index(drop=True)

                        # Ensure Symbol column exists
                        if "Symbol" not in df_copy.columns:
                            df_copy["Symbol"] = symbol

                        # Ensure Date column is properly handled
                        if (
                            "Date" not in df_copy.columns
                            and df_copy.index.name == "Date"
                        ):
                            df_copy = df_copy.reset_index()

                        # Clean any remaining index issues
                        if df_copy.index.name is not None:
                            df_copy = df_copy.reset_index(drop=True)

                        stock_dfs.append(df_copy)

                if stock_dfs:
                    combined_stock = pd.concat(stock_dfs, ignore_index=True)
                    combined_stock = combined_stock.reset_index(drop=True)
                    self.stock_data = self.preprocessor.preprocess_stock_data(
                        combined_stock
                    )
                else:
                    self.stock_data = None
            else:
                # Ensure single DataFrame has proper index
                self.stock_data = self.stock_data.reset_index(drop=True)
                self.stock_data = self.preprocessor.preprocess_stock_data(
                    self.stock_data
                )

        # Process news data
        if self.news_data is not None and not self.news_data.empty:
            self.news_data = self.preprocessor.preprocess_news_data(self.news_data)

        # Process SEC data
        if self.sec_data is not None and not self.sec_data.empty:
            self.sec_data = self.preprocessor.preprocess_sec_data(self.sec_data)

    def analyze_sentiment(self):
        """Perform sentiment analysis on textual data"""

        # Analyze news sentiment
        if self.news_data is not None and not self.news_data.empty:
            self.news_data = self.sentiment_analyzer.analyze_news_sentiment(
                self.news_data
            )

        # Analyze SEC sentiment
        if self.sec_data is not None and not self.sec_data.empty:
            self.sec_data = self.sentiment_analyzer.analyze_sec_sentiment(self.sec_data)

        # Create aggregated sentiment data
        sentiment_aggregated = self.sentiment_analyzer.aggregate_sentiment_by_date(
            self.news_data, self.sec_data
        )

        # Calculate sentiment indicators
        if not sentiment_aggregated.empty:
            sentiment_aggregated = (
                self.sentiment_analyzer.calculate_sentiment_indicators(
                    sentiment_aggregated
                )
            )

        # Save sentiment results
        self.sentiment_analyzer.save_sentiment_data(
            self.news_data if self.news_data is not None else pd.DataFrame(),
            self.sec_data if self.sec_data is not None else pd.DataFrame(),
            sentiment_aggregated,
            self.config.DATA_DIR,
        )

        self.sentiment_data = sentiment_aggregated

    def integrate_all_data(self) -> Dict[str, List[str]]:
        """Integrate all data sources"""

        # Prepare sentiment data
        sentiment_df = getattr(self, "sentiment_data", pd.DataFrame())

        # Integrate all data
        self.integrated_data = self.data_integrator.integrate_all_data(
            self.stock_data, self.news_data, self.sec_data, sentiment_df
        )

        # Create feature groups
        feature_groups = self.data_integrator.create_feature_groups(
            self.integrated_data
        )
        self.feature_groups = feature_groups

        # Save integrated data
        self.data_integrator.save_integrated_data(
            self.integrated_data, feature_groups, self.config.DATA_DIR
        )

        return feature_groups

    def train_models(self) -> Dict:
        """Train machine learning models"""

        if self.integrated_data is None or self.integrated_data.empty:
            self.logger.error("No integrated data available for training")
            return {}

        # Prepare data for modeling
        feature_groups = getattr(self, "feature_groups", None)
        X, y, feature_names = self.ml_models.prepare_modeling_data(
            self.integrated_data, feature_groups=feature_groups
        )

        if len(X) == 0:
            self.logger.error("No valid data for model training")
            return {}

        # Train models
        model_results = self.ml_models.train_models(X, y, feature_names)

        # Save models using default directory handling
        self.ml_models.save_models()

        return model_results

    def generate_predictions(self):
        """Generate risk predictions for all data"""

        if self.integrated_data is None or self.integrated_data.empty:
            self.logger.error("No data available for predictions")
            return

        # Prepare features (same as training but without target)
        exclude_cols = ["Date", "Symbol", "Risk_Label", "Risk_Score"] + [
            col
            for col in self.integrated_data.columns
            if self.integrated_data[col].dtype == "object"
        ]

        feature_cols = [
            col
            for col in self.integrated_data.columns
            if col not in exclude_cols
            and self.integrated_data[col].dtype in ["int64", "float64"]
        ]

        X = self.integrated_data[feature_cols].fillna(0)

        metadata_cols = [
            col
            for col in ["Date", "Symbol", "Returns", "Close", "Adj Close"]
            if col in self.integrated_data.columns
        ]
        metadata = (
            self.integrated_data[metadata_cols].copy() if metadata_cols else None
        )

        # Generate predictions
        predictions, probabilities = self.ml_models.predict_risk(
            X, model_type="ensemble", metadata=metadata
        )

        # Create predictions DataFrame
        import pandas as pd

        self.predictions = pd.DataFrame(
            {
                "Date": self.integrated_data["Date"],
                "Symbol": self.integrated_data["Symbol"],
                "Risk_Prediction": predictions,
                "Risk_Probability": probabilities,
                "Actual_Risk_Label": self.integrated_data.get("Risk_Label", 0),
            }
        )

        # Save predictions
        predictions_path = os.path.join(self.config.OUTPUT_DIR, "risk_predictions.csv")
        self.predictions.to_csv(predictions_path, index=False)

        self.logger.info(f"Generated {len(self.predictions)} predictions")

    def create_visualizations(self, symbols: List[str]) -> Dict[str, str]:
        """Create visualizations"""

        if self.integrated_data is None or self.integrated_data.empty:
            self.logger.error("No data available for visualization")
            return {}

        # Create comprehensive dashboard
        viz_paths = self.visualizer.create_risk_dashboard(
            self.integrated_data, symbols, self.predictions
        )

        # Get feature importance from models
        feature_importance = self.ml_models.get_feature_importance(
            "random_forest", top_n=15
        )

        # Create feature importance plot
        if feature_importance:
            importance_path = self.visualizer.plot_feature_importance(
                self.integrated_data, feature_importance
            )
            if importance_path:
                viz_paths["feature_importance"] = importance_path

        # Create summary report
        if viz_paths:
            report_path = self.visualizer.create_summary_report(viz_paths)
            viz_paths["summary_report"] = report_path

        return viz_paths

    def run_backtesting(self) -> Dict:
        """Run backtesting analysis"""

        if self.integrated_data is None or self.integrated_data.empty:
            self.logger.error("No data available for backtesting")
            return {}

        # Run comprehensive backtest
        backtest_results = self.backtester.run_comprehensive_backtest(
            self.integrated_data,
            self.predictions,
            start_date="2020-01-01",  # Focus on COVID period and after
            end_date="2023-12-31",
        )

        # Generate backtest report
        if backtest_results:
            report_path = self.backtester.generate_backtest_report(
                self.config.OUTPUT_DIR
            )
            backtest_results["report_path"] = report_path

        return backtest_results

    def generate_final_report(self, pipeline_results: Dict) -> str:
        """Generate final comprehensive report"""

        report_content = f"""
        # Market Risk Early Warning System - Final Report
        
        ## Executive Summary
        
        The Market Risk Early Warning System has been successfully executed with the following results:
        
        **Pipeline Status:** {pipeline_results.get('pipeline_status', 'Unknown')}
        **Execution Time:** {pipeline_results.get('pipeline_start', 'N/A')} to {pipeline_results.get('pipeline_end', 'N/A')}
        **Symbols Processed:** {len(pipeline_results.get('symbols_processed', []))}
        **Stages Completed:** {len(pipeline_results.get('stages_completed', []))}
        
        ## Data Collection Summary
        
        - **Stock Data:** Collected price and fundamental data from Yahoo Finance and Alpha Vantage
        - **News Data:** Gathered headlines from multiple news sources with sentiment analysis
        - **SEC Filings:** Downloaded and parsed 10-K and 10-Q filings for risk factor analysis
        
        ## Model Performance
        
        """

        model_perf = pipeline_results.get("model_performance", {})
        if model_perf:
            for model_name, metrics in model_perf.items():
                if isinstance(metrics, dict) and "auc_score" in metrics:
                    report_content += f"- **{model_name.title()}:** AUC = {metrics['auc_score']:.3f}, "
                    report_content += (
                        f"Accuracy = {metrics.get('test_accuracy', 0):.3f}\\n"
                    )

        report_content += f"""
        
        ## Feature Analysis
        
        The system analyzed {sum(pipeline_results.get('feature_groups', {}).values())} features across multiple categories:
        
        """

        for group, count in pipeline_results.get("feature_groups", {}).items():
            report_content += (
                f"- **{group.replace('_', ' ').title()}:** {count} features\\n"
            )

        report_content += f"""
        
        ## Backtesting Results
        
        The system was validated against historical market events including the COVID-19 crash, 
        Fed tightening periods, and other market stress events.
        
        """

        backtest_summary = pipeline_results.get("backtest_summary", {})
        if backtest_summary:
            data_summary = backtest_summary.get("data_summary", {})
            if data_summary:
                report_content += f"- **Total Observations:** {data_summary.get('total_observations', 0):,}\\n"
                report_content += f"- **Risk Detection Rate:** {data_summary.get('risk_distribution', {}).get('risk_rate', 0):.1%}\\n"

        report_content += f"""
        
        ## Output Files
        
        The following files have been generated:
        
        """

        viz_paths = pipeline_results.get("visualization_paths", {})
        for viz_name, path in viz_paths.items():
            report_content += f"- **{viz_name.replace('_', ' ').title()}:** `{os.path.basename(path)}`\\n"

        if pipeline_results.get("final_report_path"):
            report_content += f"- **Backtest Report:** `{os.path.basename(pipeline_results.get('backtest_summary', {}).get('report_path', ''))}`\\n"

        report_content += f"""
        
        ## Next Steps
        
        1. Review the generated visualizations to understand risk patterns
        2. Analyze the backtest report to validate system performance
        3. Consider adjusting model parameters based on results
        4. Deploy the system for real-time monitoring (requires additional infrastructure)
        
        ## Technical Notes
        
        - All models use free-tier data sources to minimize costs
        - System is designed to run on local machines or Google Colab
        - Rate limiting is implemented to respect API constraints
        - Modular design allows for easy enhancement and modification
        
        ---
        
        Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

        # Save report
        report_path = os.path.join(self.config.OUTPUT_DIR, "final_report.md")
        with open(report_path, "w") as f:
            f.write(report_content)

        self.logger.info(f"Generated final report: {report_path}")
        return report_path

    def generate_research_artifacts(self) -> Optional[Dict[str, Any]]:
        """Create research-grade evaluation, hypothesis, and robustness outputs."""

        if self.predictions is None or self.predictions.empty:
            self.logger.warning("No predictions available for research artifacts")
            return None

        if self.integrated_data is None or self.integrated_data.empty:
            self.logger.warning("Integrated data missing - skipping research artifacts")
            return None

        evaluation_df = self.predictions.copy()
        if "Actual_Risk_Label" in evaluation_df.columns:
            evaluation_df.rename(
                columns={"Actual_Risk_Label": "Risk_Label"}, inplace=True
            )
        if "Risk_Label" not in evaluation_df.columns:
            self.logger.warning("Risk labels missing for evaluation")
            return None

        evaluator = ResearchEvaluator()
        evaluation_results = evaluator.evaluate_predictions(
            evaluation_df,
            label_col="Risk_Label",
            probability_col="Risk_Probability",
        )

        feature_groups = getattr(self, "feature_groups", {}) or {}
        fundamental_keys = [
            "price_features",
            "volume_features",
            "technical_indicators",
            "fundamental_ratios",
            "market_features",
        ]
        fundamental_features: List[str] = []
        for key in fundamental_keys:
            fundamental_features.extend(feature_groups.get(key, []))

        sentiment_keys = ["sentiment_features", "news_features", "sec_features"]
        sentiment_features: List[str] = []
        for key in sentiment_keys:
            sentiment_features.extend(feature_groups.get(key, []))
        if not sentiment_features:
            sentiment_features = [
                col
                for col in self.integrated_data.columns
                if "sentiment" in col.lower() or col.startswith("news_embedding_")
            ]

        hypothesis_results: Dict[str, Any] = {}
        if fundamental_features and sentiment_features:
            tester = SentimentImpactTester()
            merged_df = self.integrated_data.copy()
            if "Risk_Label" not in merged_df.columns:
                merged_df["Risk_Label"] = evaluation_df["Risk_Label"].values
            try:
                sentiment_result = tester.run_test(
                    merged_df,
                    fundamental_features=[
                        feat
                        for feat in fundamental_features
                        if feat in merged_df.columns
                    ],
                    sentiment_features=[
                        feat
                        for feat in sentiment_features
                        if feat in merged_df.columns
                    ],
                )
                hypothesis_results["sentiment_vs_fundamentals"] = sentiment_result.__dict__
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning("Hypothesis test failed: %s", exc)

        graph_ablation = GraphFeatureAblation()
        ablation_df = self.integrated_data.copy()
        if "Risk_Label" not in ablation_df.columns:
            ablation_df["Risk_Label"] = evaluation_df["Risk_Label"].values
        try:
            hypothesis_results["graph_feature_ablation"] = graph_ablation.evaluate(
                ablation_df
            )
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Graph ablation failed: %s", exc)

        robustness_results: Dict[str, Any] = {}
        if self.news_data is not None and not self.news_data.empty:
            sentiment_col = "sentiment_score"
            group_col = "Source"
            if sentiment_col in self.news_data.columns:
                if group_col not in self.news_data.columns:
                    group_col = "Symbol"
                if group_col in self.news_data.columns:
                    top_groups = (
                        self.news_data[group_col]
                        .value_counts()
                        .index.tolist()
                    )
                    if len(top_groups) >= 2:
                        detector = SentimentBiasDetector(sentiment_col=sentiment_col)
                        try:
                            bias_report = detector.compare_groups(
                                self.news_data,
                                group_col=group_col,
                                group_a=top_groups[0],
                                group_b=top_groups[1],
                            )
                            robustness_results["sentiment_bias"] = bias_report.__dict__
                        except Exception as exc:  # pragma: no cover
                            self.logger.warning("Bias detection failed: %s", exc)

        if self.ml_models.feature_names:
            stress_tester = RobustnessStressTester()
            feature_cols = [
                col for col in self.ml_models.feature_names if col in self.integrated_data.columns
            ]
            if feature_cols:
                base_features = self.integrated_data[feature_cols].fillna(0)
                metadata_cols = [
                    col
                    for col in ["Date", "Symbol", "Returns", "Close", "Adj Close"]
                    if col in self.integrated_data.columns
                ]
                metadata = (
                    self.integrated_data[metadata_cols].copy() if metadata_cols else None
                )
                _, base_probs = self.ml_models.predict_risk(
                    base_features, metadata=metadata
                )
                noisy_features = stress_tester.inject_noise(
                    base_features,
                    noise_level=0.2,
                    columns=feature_cols,
                )
                _, noisy_probs = self.ml_models.predict_risk(
                    noisy_features, metadata=metadata
                )
                labels = evaluation_df["Risk_Label"].to_numpy()
                robustness_results["adversarial_noise"] = {
                    "baseline": evaluator.benchmarks.evaluate_metrics(
                        labels, base_probs
                    ),
                    "noisy": evaluator.benchmarks.evaluate_metrics(
                        labels, noisy_probs
                    ),
                }

        report_builder = ResearchReportBuilder()
        markdown_path = report_builder.build_markdown(
            evaluation_results=evaluation_results,
            hypothesis_results=hypothesis_results,
            robustness_results=robustness_results,
        )
        html_path = report_builder.build_html(
            evaluation_results=evaluation_results,
            hypothesis_results=hypothesis_results,
            robustness_results=robustness_results,
        )

        artifact_summary = {
            "evaluation_results": evaluation_results,
            "hypothesis_results": hypothesis_results,
            "robustness_results": robustness_results,
            "report_markdown": str(markdown_path),
            "report_html": str(html_path),
        }

        self.logger.info(
            "Research artifacts generated: %s and %s",
            markdown_path,
            html_path,
        )
        return artifact_summary


def create_cli_parser():
    """Create command line interface"""

    parser = argparse.ArgumentParser(
        description="Market Risk Early Warning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --full-pipeline                    # Run complete pipeline
  python main.py --symbols AAPL MSFT GOOGL         # Analyze specific symbols
  python main.py --backtest-only                    # Run backtesting only
  python main.py --start-date 2020-01-01 --end-date 2023-12-31  # Custom date range
        """,
    )

    # Main actions
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run the complete pipeline from data collection to analysis",
    )
    parser.add_argument(
        "--backtest-only",
        action="store_true",
        help="Run backtesting analysis only (requires existing data)",
    )
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Create visualizations only (requires existing data)",
    )

    # Data options
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Stock symbols to analyze (e.g., AAPL MSFT GOOGL)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for data collection (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for data collection (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--skip-data-collection",
        action="store_true",
        help="Skip data collection and use existing data",
    )

    # Output options
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for results"
    )
    parser.add_argument(
        "--config-file", type=str, default=None, help="Path to configuration file"
    )

    return parser


def main():
    """Main entry point"""

    parser = create_cli_parser()
    args = parser.parse_args()

    # Initialize system
    try:
        system = MarketRiskSystem(args.config_file)
    except Exception as e:
        print(f"Failed to initialize system: {e}")
        sys.exit(1)

    # Override output directory if specified
    if args.output_dir:
        system.config.OUTPUT_DIR = args.output_dir
        system.visualizer.output_dir = args.output_dir

    try:
        if args.full_pipeline:
            # Run complete pipeline
            results = system.run_full_pipeline(
                symbols=args.symbols,
                start_date=args.start_date,
                end_date=args.end_date,
                skip_data_collection=args.skip_data_collection,
            )

            print("\\n" + "=" * 60)
            print("PIPELINE EXECUTION SUMMARY")
            print("=" * 60)
            print(f"Status: {results.get('pipeline_status', 'Unknown')}")
            print(f"Stages Completed: {len(results.get('stages_completed', []))}")
            print(f"Symbols Processed: {len(results.get('symbols_processed', []))}")

            if results.get("errors"):
                print(f"Errors: {len(results['errors'])}")
                for error in results["errors"]:
                    print(f"  - {error}")

            if results.get("final_report_path"):
                print(f"\\nFinal Report: {results['final_report_path']}")

            if results.get("visualization_paths"):
                print(
                    f"\\nVisualizations created: {len(results['visualization_paths'])}"
                )
                for name, path in results["visualization_paths"].items():
                    print(f"  - {name}: {path}")

        elif args.backtest_only:
            # Load existing data and run backtesting
            system.load_existing_data()
            system.preprocess_all_data()
            system.analyze_sentiment()
            feature_groups = system.integrate_all_data()

            if system.integrated_data is not None:
                backtest_results = system.run_backtesting()
                print(
                    f"\\nBacktest completed. Report: {backtest_results.get('report_path', 'N/A')}"
                )
            else:
                print("No data available for backtesting")

        elif args.visualize_only:
            # Load existing data and create visualizations
            system.load_existing_data()
            system.preprocess_all_data()
            system.analyze_sentiment()
            feature_groups = system.integrate_all_data()

            if system.integrated_data is not None:
                viz_paths = system.create_visualizations(
                    args.symbols or ["AAPL", "MSFT", "GOOGL"]
                )
                print(f"\\nVisualizations created: {len(viz_paths)}")
                for name, path in viz_paths.items():
                    print(f"  - {name}: {path}")
            else:
                print("No data available for visualization")

        else:
            # Show help if no action specified
            parser.print_help()

    except KeyboardInterrupt:
        print("\\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\nExecution failed: {e}")
        logging.exception("Detailed error information:")
        sys.exit(1)


if __name__ == "__main__":
    main()
