"""
Configuration management for the Market Risk Early Warning System
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from src.utils.logging import configure_logging, get_logger

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the early warning system"""

    # API Configuration
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")

    # Data Configuration
    START_DATE = os.getenv("START_DATE", "2021-01-01")
    END_DATE = os.getenv("END_DATE", "2023-12-31")
    LOOKBACK_YEARS = int(os.getenv("LOOKBACK_YEARS", 2))

    # Model Configuration
    RISK_THRESHOLD = float(os.getenv("RISK_THRESHOLD", 0.7))
    VOLATILITY_WINDOW = int(os.getenv("VOLATILITY_WINDOW", 30))
    SENTIMENT_WINDOW = int(os.getenv("SENTIMENT_WINDOW", 7))

    # Directories
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    MLFLOW_TRACKING_DIR = os.path.join(OUTPUT_DIR, "mlruns")
    MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "MEWS-Experiment")

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # S&P 500 companies (top 50 for free tier limits)
    SP500_SYMBOLS = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "TSLA",
        "META",
        "BRK-B",
        "UNH",
        "JNJ",
        "JPM",
        "V",
        "PG",
        "XOM",
        "HD",
        "CVX",
        "MA",
        "BAC",
        "ABBV",
        "PFE",
        "AVGO",
        "KO",
        "LLY",
        "WMT",
        "MRK",
        "PEP",
        "COST",
        "TMO",
        "DHR",
        "NEE",
        "VZ",
        "ABT",
        "ADBE",
        "NFLX",
        "CRM",
        "CMCSA",
        "ACN",
        "INTC",
        "NKE",
        "TXN",
        "AMD",
        "QCOM",
        "UPS",
        "PM",
        "HON",
        "T",
        "RTX",
        "LOW",
        "SPGI",
        "IBM",
    ]

    # API Rate Limits (requests per minute)
    ALPHA_VANTAGE_RATE_LIMIT = 5
    GNEWS_RATE_LIMIT = 10
    SEC_EDGAR_RATE_LIMIT = 10
    YAHOO_FINANCE_RATE_LIMIT = 2000  # Very generous, but be respectful

    @classmethod
    def setup_logging(cls):
        """Setup logging configuration"""
        configure_logging(
            level=cls.LOG_LEVEL,
            output_dir=cls.OUTPUT_DIR,
            filename="risk_system.log",
            force=True,
        )

    @classmethod
    def setup_mlflow(cls):
        """Configure MLflow tracking to use the project outputs directory."""

        try:
            import mlflow  # type: ignore

            tracking_dir = Path(cls.MLFLOW_TRACKING_DIR)
            tracking_dir.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
            mlflow.set_experiment(cls.MLFLOW_EXPERIMENT)
            get_logger(__name__).info(
                "MLflow tracking set to %s for experiment '%s'",
                tracking_dir,
                cls.MLFLOW_EXPERIMENT,
            )
        except ImportError:
            get_logger(__name__).warning(
                "MLflow is not installed; experiment tracking is disabled."
            )

    @classmethod
    def validate_config(cls):
        """Validate configuration and API keys"""
        missing_keys = []

        if not cls.ALPHA_VANTAGE_API_KEY:
            missing_keys.append("ALPHA_VANTAGE_API_KEY")

        if not cls.GNEWS_API_KEY:
            missing_keys.append("GNEWS_API_KEY")

        if missing_keys:
            get_logger(__name__).warning(
                f"Missing API keys: {missing_keys}. Some features may not work."
            )

        return len(missing_keys) == 0

    @classmethod
    def get_date_range(cls):
        """Get start and end dates as datetime objects"""
        start_date = datetime.strptime(cls.START_DATE, "%Y-%m-%d")
        end_date = datetime.strptime(cls.END_DATE, "%Y-%m-%d")
        return start_date, end_date
