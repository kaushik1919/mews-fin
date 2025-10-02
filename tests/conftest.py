"""
Pytest configuration and fixtures.
"""

import os
import sys
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(scope="session")
def sample_stock_data():
    """Create sample stock data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")

    data = {
        "Open": np.random.uniform(90, 110, 100),
        "High": np.random.uniform(95, 115, 100),
        "Low": np.random.uniform(85, 105, 100),
        "Close": np.random.uniform(90, 110, 100),
        "Volume": np.random.randint(500000, 2000000, 100),
    }

    df = pd.DataFrame(data, index=dates)

    # Make close prices more realistic with some correlation
    for i in range(1, len(df)):
        df.iloc[i]["Close"] = df.iloc[i - 1]["Close"] * (1 + np.random.normal(0, 0.02))

    return df


@pytest.fixture(scope="session")
def sample_news_data():
    """Create sample news data for testing."""
    return [
        {
            "title": "Market shows strong performance with tech stocks leading",
            "description": "Technology sector drives market gains",
            "published_date": "2023-01-01",
            "source": "Financial News",
        },
        {
            "title": "Economic uncertainty creates market volatility",
            "description": "Investors cautious amid economic indicators",
            "published_date": "2023-01-02",
            "source": "Market Watch",
        },
        {
            "title": "Federal Reserve maintains interest rate policy",
            "description": "Central bank keeps rates steady for economic stability",
            "published_date": "2023-01-03",
            "source": "Economic Times",
        },
    ]


@pytest.fixture
def mock_api_response():
    """Mock API response for testing."""
    return Mock()


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary directory for test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def temp_models_dir(tmp_path):
    """Create temporary directory for test models."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return models_dir
