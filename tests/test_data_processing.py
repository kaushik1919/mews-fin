"""
Unit tests for data processing modules.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_fetcher import DataFetcher
from data_preprocessor import DataPreprocessor
from sentiment_analyzer import SentimentAnalyzer


class TestDataFetcher:
    """Test cases for DataFetcher class."""
    
    @pytest.fixture
    def data_fetcher(self):
        """Create DataFetcher instance."""
        return DataFetcher()
    
    def test_data_fetcher_initialization(self, data_fetcher):
        """Test DataFetcher initialization."""
        assert data_fetcher is not None
        assert hasattr(data_fetcher, 'fetch_stock_data')
    
    @patch('data_fetcher.yf.download')
    def test_fetch_stock_data_success(self, mock_yf_download, data_fetcher):
        """Test successful stock data fetching."""
        # Mock yfinance response
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        mock_yf_download.return_value = mock_data
        
        result = data_fetcher.fetch_stock_data(['AAPL'], '2023-01-01', '2023-01-03')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'Close' in result.columns
        assert 'Volume' in result.columns
    
    @patch('data_fetcher.yf.download')
    def test_fetch_stock_data_failure(self, mock_yf_download, data_fetcher):
        """Test handling of data fetching failure."""
        mock_yf_download.side_effect = Exception("Network error")
        
        with pytest.raises(Exception):
            data_fetcher.fetch_stock_data(['AAPL'], '2023-01-01', '2023-01-03')


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    @pytest.fixture
    def sample_stock_data(self):
        """Create sample stock data."""
        return pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [103, 104, 105, 106, 107],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=pd.date_range('2023-01-01', periods=5))
    
    @pytest.fixture
    def preprocessor(self):
        """Create DataPreprocessor instance."""
        return DataPreprocessor()
    
    def test_calculate_technical_indicators(self, preprocessor, sample_stock_data):
        """Test technical indicator calculations."""
        result = preprocessor.calculate_technical_indicators(sample_stock_data)
        
        assert 'rsi' in result.columns
        assert 'sma_20' in result.columns
        assert 'price_change' in result.columns
        assert len(result) == len(sample_stock_data)
    
    def test_create_risk_labels(self, preprocessor, sample_stock_data):
        """Test risk label creation."""
        # Add price changes that should trigger risk labels
        data_with_changes = sample_stock_data.copy()
        data_with_changes.loc['2023-01-03', 'Close'] = 90  # Large drop
        
        result = preprocessor.create_risk_labels(data_with_changes, threshold=0.05)
        
        assert 'risk_label' in result.columns
        assert result['risk_label'].dtype == int
        assert set(result['risk_label'].unique()) <= {0, 1}
    
    def test_handle_missing_values(self, preprocessor):
        """Test missing value handling."""
        data_with_nan = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [np.nan, 2, 3, 4, 5],
            'feature3': [1, 2, 3, 4, np.nan]
        })
        
        result = preprocessor.handle_missing_values(data_with_nan)
        
        assert not result.isnull().any().any()
        assert len(result) == len(data_with_nan)


class TestSentimentAnalyzer:
    """Test cases for SentimentAnalyzer class."""
    
    @pytest.fixture
    def sentiment_analyzer(self):
        """Create SentimentAnalyzer instance."""
        return SentimentAnalyzer()
    
    def test_analyzer_initialization(self, sentiment_analyzer):
        """Test SentimentAnalyzer initialization."""
        assert sentiment_analyzer is not None
        assert hasattr(sentiment_analyzer, 'analyze_sentiment')
    
    def test_analyze_positive_sentiment(self, sentiment_analyzer):
        """Test positive sentiment analysis."""
        positive_text = "The stock market is performing excellently with great returns!"
        
        result = sentiment_analyzer.analyze_sentiment(positive_text)
        
        assert isinstance(result, dict)
        assert 'compound' in result
        assert result['compound'] > 0
    
    def test_analyze_negative_sentiment(self, sentiment_analyzer):
        """Test negative sentiment analysis."""
        negative_text = "The market is crashing terribly with massive losses!"
        
        result = sentiment_analyzer.analyze_sentiment(negative_text)
        
        assert isinstance(result, dict)
        assert 'compound' in result
        assert result['compound'] < 0
    
    def test_analyze_neutral_sentiment(self, sentiment_analyzer):
        """Test neutral sentiment analysis."""
        neutral_text = "The market opened today at normal levels."
        
        result = sentiment_analyzer.analyze_sentiment(neutral_text)
        
        assert isinstance(result, dict)
        assert 'compound' in result
        assert abs(result['compound']) < 0.5
    
    def test_batch_sentiment_analysis(self, sentiment_analyzer):
        """Test batch sentiment analysis."""
        texts = [
            "Great market performance!",
            "Terrible market crash!",
            "Market opened normally."
        ]
        
        results = [sentiment_analyzer.analyze_sentiment(text) for text in texts]
        
        assert len(results) == 3
        assert all(isinstance(result, dict) for result in results)
        assert all('compound' in result for result in results)
    
    def test_empty_text_handling(self, sentiment_analyzer):
        """Test handling of empty text."""
        result = sentiment_analyzer.analyze_sentiment("")
        
        assert isinstance(result, dict)
        assert 'compound' in result
        assert result['compound'] == 0.0


@pytest.mark.integration
class TestDataProcessingIntegration:
    """Integration tests for data processing pipeline."""
    
    def test_full_data_pipeline(self):
        """Test complete data processing pipeline."""
        # Create mock data
        stock_data = pd.DataFrame({
            'Open': np.random.uniform(90, 110, 50),
            'High': np.random.uniform(95, 115, 50),
            'Low': np.random.uniform(85, 105, 50),
            'Close': np.random.uniform(90, 110, 50),
            'Volume': np.random.randint(500000, 2000000, 50)
        }, index=pd.date_range('2023-01-01', periods=50))
        
        # Add some realistic price movements
        for i in range(1, len(stock_data)):
            stock_data.iloc[i]['Close'] = (
                stock_data.iloc[i-1]['Close'] * 
                (1 + np.random.normal(0, 0.02))
            )
        
        preprocessor = DataPreprocessor()
        
        # Process data
        processed_data = preprocessor.calculate_technical_indicators(stock_data)
        processed_data = preprocessor.create_risk_labels(processed_data)
        processed_data = preprocessor.handle_missing_values(processed_data)
        
        # Verify pipeline results
        assert len(processed_data) > 0
        assert 'risk_label' in processed_data.columns
        assert not processed_data.isnull().any().any()
        assert processed_data['risk_label'].dtype == int
        
        # Test sentiment integration
        sample_news = [
            "Market shows strong growth",
            "Economic indicators look positive",
            "Investors remain optimistic"
        ]
        
        sentiment_analyzer = SentimentAnalyzer()
        sentiments = [sentiment_analyzer.analyze_sentiment(news) 
                     for news in sample_news]
        
        # Calculate average sentiment
        avg_sentiment = np.mean([s['compound'] for s in sentiments])
        
        # Add to processed data
        processed_data['sentiment_score'] = avg_sentiment
        
        assert 'sentiment_score' in processed_data.columns
        assert -1 <= processed_data['sentiment_score'].iloc[0] <= 1
