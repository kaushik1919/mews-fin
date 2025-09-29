# 📊 Market Risk Early Warning System (MEWS)

A professional-grade machine learning system for predicting market risk using free data sources. Features GPU acceleration, real-time sentiment analysis, and an interactive web dashboard.

## ✨ Features

- 🤖 **4 ML Models**: Random Forest, XGBoost, SVM, and Logistic Regression with ensemble predictions
- 🚀 **GPU Acceleration**: CUDA support for faster training (optional)
- 📰 **Real-time Sentiment**: News sentiment analysis with GNews API integration
- 📊 **Interactive Dashboard**: Professional Streamlit web interface
- 🔄 **CI/CD Pipeline**: Automated testing, linting, and deployment
- 🐳 **Docker Support**: Containerized deployment for easy scaling
- 📈 **Risk Timeline**: Historical risk visualization with statistical analysis

## 🚀 Quick Start

### Option 1: Using Build Script (Recommended)

```bash
# Clone repository
git clone https://github.com/kaushik1919/mews-fin.git
cd mews-fin

# Set up development environment (includes testing tools)
python build.py setup-dev

# Run the web application
streamlit run streamlit_app.py
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"

# Run application
streamlit run streamlit_app.py
```

### Option 3: Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d mews-app

# Access at http://localhost:8501
```

## ⚙️ Configuration

Copy `.env.example` to `.env` and configure your API keys:

```bash
cp .env.example .env
```

Required API keys (all have free tiers):
- **GNews API** ([Get key](https://gnews.io/)): 100 requests/day
- Optional: Alpha Vantage, News API, FRED API

## 🧪 Development & Testing

```bash
# Install development dependencies
python build.py setup-dev

# Run tests with coverage
python build.py test

# Run code quality checks
python build.py lint

# Format code
python build.py format

# Run everything
python build.py all
```

## 📊 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Processing    │    │   ML & Output   │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ Yahoo Finance   │───▶│ Data Cleaning   │───▶│ ML Models       │
│ Alpha Vantage   │    │ Feature Eng.    │    │ Risk Prediction │
│ SEC EDGAR       │───▶│ Sentiment Anal. │───▶│ Visualization   │
│ News APIs       │    │ Data Integration│    │ Backtesting     │
│ Market Data     │    │ Risk Labeling   │    │ Reports         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 System Components

### Core Modules

1. **`config.py`**: Configuration management and API key handling
2. **`data_fetcher.py`**: Stock data collection from Yahoo Finance/Alpha Vantage
3. **`sec_downloader.py`**: SEC filings download and parsing
4. **`news_collector.py`**: Multi-source news collection and processing
5. **`data_preprocessor.py`**: Data cleaning and feature engineering
6. **`sentiment_analyzer.py`**: VADER and FinBERT sentiment analysis
7. **`data_integrator.py`**: Multi-source data integration and feature creation
8. **`ml_models.py`**: Random Forest and Logistic Regression models
9. **`visualizer.py`**: Interactive dashboards and static plots
10. **`backtester.py`**: Historical validation and performance metrics
11. **`main.py`**: Main orchestrator with CLI interface

### Key Features

- **Free Data Sources**: Optimized for free-tier APIs
- **Lightweight Compute**: Runs on Colab, local machines, or lightweight cloud instances  
- **Modular Design**: Each component can be used independently
- **Comprehensive Backtesting**: Validates against COVID crash and other market events
- **Interactive Visualizations**: Plotly dashboards and matplotlib charts
- **Risk Prediction**: Binary classification with probability scores
- **Rate Limiting**: Respects API constraints automatically

## 📈 Example Outputs

The system generates:

### Visualizations
- Risk timeline charts
- Sentiment analysis plots  
- Feature importance rankings
- Model performance metrics
- Interactive dashboards

### Reports
- Backtest performance analysis
- Prediction accuracy metrics
- Market event detection results
- Feature correlation analysis
- Comprehensive final report

### Data Files
- `integrated_data.csv`: Combined features from all sources
- `risk_predictions.csv`: Model predictions with probabilities
- `sentiment_aggregated.csv`: Daily sentiment scores
- Model files (`.joblib` format)

## 🎯 Model Performance

The system includes two main models:

### Random Forest
- **Strengths**: Handles non-linear patterns, robust to outliers
- **Use Case**: Complex feature interactions, ensemble predictions

### Logistic Regression  
- **Strengths**: Interpretable, fast training, good baseline
- **Use Case**: Linear relationships, feature importance analysis

### Performance Metrics
- AUC-ROC score
- Precision/Recall
- Confusion matrix
- Feature importance
- Cross-validation scores

## 🔍 Backtesting

The system validates predictions against major market events:

### COVID-19 Crash (Feb-Mar 2020)
- Market drop: 35% in 5 weeks
- Validation: Early warning detection
- Metrics: Precision/recall during crisis

### Federal Reserve Tightening (2022)
- Interest rate increases
- Market volatility patterns
- Sector rotation analysis

### Banking Crisis (Mar 2023)
- Regional bank stress
- Systemic risk indicators
- Contagion detection

## 🛠️ Development

### Project Structure
```
mews-fin/
├── src/                 # Core modules
├── data/                # Raw and processed data
├── models/              # Trained ML models
├── outputs/             # Visualizations and reports
├── config/              # Configuration files
├── notebooks/           # Jupyter notebooks (optional)
├── main.py              # Main orchestrator
├── requirements.txt     # Dependencies
└── README.md            # This file
```

### Adding New Features

1. **New Data Source**: Extend `data_fetcher.py` or create new collector
2. **New Model**: Add to `ml_models.py` model registry
3. **New Visualization**: Extend `visualizer.py` dashboard
4. **New Indicator**: Add to `data_preprocessor.py` feature engineering

### Testing

```bash
# Run individual components
python -m src.data_fetcher
python -m src.sentiment_analyzer
python -m src.ml_models

# Test with sample data
python main.py --symbols AAPL --start-date 2023-01-01 --end-date 2023-06-30
```

## 📚 Documentation

### Key Parameters

- **Risk Threshold**: Market drops >5% trigger risk labels
- **Lookback Window**: 30-day rolling features
- **Model Retraining**: Monthly or quarterly
- **Sentiment Sources**: News headlines, SEC filings

### API Rate Limits

- **Alpha Vantage**: 5 calls/minute (free tier)
- **Yahoo Finance**: No official limits (use responsibly)
- **SEC EDGAR**: 10 requests/second
- **News APIs**: Vary by provider (100-1000 requests/day)

### Memory Requirements

- **Minimum**: 4GB RAM for basic analysis
- **Recommended**: 8GB RAM for full pipeline
- **Storage**: ~1GB for 5 years of data (50 stocks)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This system is for educational and research purposes only. Not intended for live trading or investment advice. Always conduct your own research and consult with financial advisors before making investment decisions.

## 🆘 Support

For issues and questions:
1. Check existing GitHub issues
2. Create new issue with detailed description
3. Include error logs and system configuration

---

