"""
Unit tests for data processing modules.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from src.data_fetcher import StockDataFetcher
from src.data_preprocessor import DataPreprocessor
from src.sentiment_analyzer import SentimentAnalyzer


class TestStockDataFetcher:
    """Test cases for :class:`StockDataFetcher`."""
    
    @pytest.fixture
    def data_fetcher(self):
        """Create StockDataFetcher instance."""
        return StockDataFetcher()
    
    def test_data_fetcher_initialization(self, data_fetcher):
        """Test DataFetcher initialization."""
        assert data_fetcher is not None
        assert hasattr(data_fetcher, "fetch_yahoo_data")
    
    @patch("src.data_fetcher.yf.Ticker")
    def test_fetch_stock_data_success(self, mock_yf_ticker, data_fetcher):
        """Test successful stock data fetching."""
        # Mock yfinance response
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_data
        mock_ticker.info = {}
        mock_yf_ticker.return_value = mock_ticker

        result = data_fetcher.fetch_yahoo_data(['AAPL'], '2023-01-01', '2023-01-03')

        assert 'AAPL' in result
        fetched_df = result['AAPL']
        assert isinstance(fetched_df, pd.DataFrame)
        assert len(fetched_df) == 3
        assert 'Close' in fetched_df.columns
        assert 'Volume' in fetched_df.columns
        assert (fetched_df['Symbol'] == 'AAPL').all()

    @patch("src.data_fetcher.yf.Ticker")
    def test_fetch_stock_data_failure(self, mock_yf_ticker, data_fetcher):
        """Test handling of data fetching failure."""
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = Exception("Network error")
        mock_yf_ticker.return_value = mock_ticker
        
        result = data_fetcher.fetch_yahoo_data(['AAPL'], '2023-01-01', '2023-01-03')

        assert 'AAPL' not in result


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    @pytest.fixture
    def sample_stock_data(self):
        """Create sample stock data."""
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=5),
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [103, 104, 105, 106, 107],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000],
            'Symbol': ['AAPL'] * 5
        })
        return df
    
    @pytest.fixture
    def preprocessor(self):
        """Create DataPreprocessor instance."""
        return DataPreprocessor()
    
    def test_preprocess_stock_data_adds_features(self, preprocessor, sample_stock_data):
        """Technical indicators should be added during preprocessing."""
        result = preprocessor.preprocess_stock_data(sample_stock_data)

        assert 'Returns' in result.columns
        assert 'RSI' in result.columns
        assert 'MACD' in result.columns
        assert len(result) == len(sample_stock_data)
    
    def test_create_risk_labels(self, preprocessor, sample_stock_data):
        """Test risk label creation."""
        # Add price changes that should trigger risk labels
        data_with_changes = sample_stock_data.copy()
        data_with_changes.loc[2, 'Close'] = 90  # Large drop

        processed = preprocessor.preprocess_stock_data(data_with_changes)
        result = preprocessor.create_risk_labels(
            processed,
            volatility_threshold=0.01,
            return_threshold=-0.02,
            window=2,
        )

        assert 'Risk_Label' in result.columns
        assert set(result['Risk_Label'].unique()) <= {0, 1}
    
    def test_handle_missing_values(self, preprocessor):
        """Test missing value handling."""
        data_with_nan = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=5),
            'Open': [100, np.nan, 102, np.nan, 104],
            'High': [105, 106, np.nan, 108, 109],
            'Low': [95, 96, 97, np.nan, 99],
            'Close': [103, np.nan, 105, 106, 107],
            'Volume': [1000000, 1100000, np.nan, 1300000, 1400000],
            'Symbol': ['AAPL'] * 5
        })

        result = preprocessor.preprocess_stock_data(data_with_nan)

        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        assert not result[numeric_cols].isnull().any().any()


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
            'Date': pd.date_range('2023-01-01', periods=50),
            'Open': np.random.uniform(90, 110, 50),
            'High': np.random.uniform(95, 115, 50),
            'Low': np.random.uniform(85, 105, 50),
            'Close': np.random.uniform(90, 110, 50),
            'Volume': np.random.randint(500000, 2000000, 50),
            'Symbol': ['AAPL'] * 50
        })
        
        # Add some realistic price movements
        for i in range(1, len(stock_data)):
            stock_data.loc[i, 'Close'] = (
                stock_data.loc[i - 1, 'Close'] *
                (1 + np.random.normal(0, 0.02))
            )
        
        preprocessor = DataPreprocessor()
        
        # Process data
        processed_data = preprocessor.preprocess_stock_data(stock_data)
        processed_data = preprocessor.create_risk_labels(processed_data)
        processed_data = processed_data.dropna(subset=['Risk_Label'])
        
        # Verify pipeline results
        assert len(processed_data) > 0
        assert 'Risk_Label' in processed_data.columns
        assert not processed_data.isnull().any().any()
        assert set(processed_data['Risk_Label'].unique()) <= {0, 1}
        
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
