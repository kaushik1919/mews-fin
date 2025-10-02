# 📊 Market Risk Early Warning System (MEWS)

A professional-grade machine learning system for predicting market risk using free data sources. Features GPU acceleration, real-time sentiment analysis, and an interactive web dashboard.

> **Research framing:** MEWS is designed as a reproducible research stack that fuses market microstructure data, textual sentiment, and correlation graphs to stress-test systemic risk hypotheses. The pipeline supports crisis replays, ablation experiments, and robustness diagnostics so you can defend findings with publication-ready evidence.

## ✨ Features

- 🤖 **4 ML Models**: Random Forest, XGBoost, SVM, and Logistic Regression with ensemble predictions
- 🚀 **GPU Acceleration**: CUDA support for faster training (optional)
- 📰 **Real-time Sentiment**: News sentiment analysis with GNews API integration
- 📊 **Interactive Dashboard**: Professional Streamlit web interface
- 🔄 **CI/CD Pipeline**: Automated testing, linting, and deployment
- 🐳 **Docker Support**: Containerized deployment for easy scaling
- 📈 **Risk Timeline**: Historical risk visualization with statistical analysis
- 🕸️ **Graph-Aware Risk Signals**: Correlation GNN features powered by PyTorch Geometric

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

# If pip cannot resolve torch-geometric automatically, follow the
# [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

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

### Run the full research pipeline

```powershell
python main.py --full-pipeline
```

### Launch the Streamlit dashboard

```powershell
streamlit run streamlit_app.py
```

## 🖥️ Command-Line Interface

Once the dependencies are installed (via `python build.py setup-dev`, `pip install -r requirements.txt`, or `pip install -e .`), you can drive the full system directly from the command line. The CLI is available either through the module entry point or the installed console script.

- **Show all available switches:**
  - `python main.py --help`
  - `mews-fin --help`
- **Run the end-to-end pipeline with data collection, modeling, reporting, and research addendum artifacts:**
  - `python main.py --full-pipeline`
  - `mews-fin --full-pipeline`
- **Re-run analytics against previously collected data without fetching again:**
  - `python main.py --full-pipeline --skip-data-collection`
  - `mews-fin --full-pipeline --skip-data-collection`
- **Execute just the backtesting stage (expects preprocessed data on disk):**
  - `python main.py --backtest-only --skip-data-collection`
  - `mews-fin --backtest-only --skip-data-collection`

On Windows PowerShell, prefix virtual-environment scripts with `.\` (for example `.\venv\Scripts\activate`). The console script `mews-fin` and its alias `mews-fin-cli` are provided via the project metadata in `pyproject.toml` for installations performed with `pip install .` or `pip install -e .`.

When `--full-pipeline` completes, the system runs a tenth "Research Addendum" stage that benchmarks MEWS against statistical baselines, performs hypothesis tests, evaluates robustness, and publishes Markdown/HTML summaries inside `outputs/research/`. See `docs/RESEARCH_GUIDE.md` for interpretation tips and citation-ready figures.

## ⚙️ Configuration

Copy the provided template to a root-level `.env` and confirm your keys:

```bash
cp config/.env.template .env
```

The template already carries a working demo `GNEWS_API_KEY` (from the `config/.env.template` attachment above). Update any other keys you plan to use:

```ini
# .env
GNEWS_API_KEY=your_gnews_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
NEWS_API_KEY=optional_news_api_key
```

If you prefer to keep secrets outside the repo, export them as environment variables instead of editing `.env`.

Required API keys (all have free tiers):
- **GNews API** ([Get key](https://gnews.io/)): 100 requests/day (already seeded in the template)
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
12. **`graph_models/gnn_risk_predictor.py`**: Correlation-graph GCN/GAT that produces node-level risk scores for the ensemble

### Baseline Benchmarks & Experiments

- **`src/baselines/`** now includes statistical and deep learning baselines:
  - `GARCHBaseline`: conditional volatility with one-step Value-at-Risk thresholds
  - `ValueAtRiskBaseline`: historical simulation + parametric normal VaR
  - `LSTMBaseline`: sequence model returning per-date risk probabilities
- **`src/experiments/experiment_manager.py`** orchestrates configurable experiments, ablations, and compares MEWS against the baselines. It saves every run under `outputs/experiments/<timestamp>_<name>/` with per-model CSVs, summary JSON, and optional MEWS predictions.

Create a lightweight experiment by supplying a Pandas dataframe or a YAML/JSON config:

```python
from src.experiments import ExperimentConfig, ExperimentManager

config = ExperimentConfig.from_mapping({
    "name": "baseline_demo",
    "data_path": "data/integrated_dataset.csv",
    "baselines": [
        {"type": "garch", "params": {"confidence": 0.95}},
        {"type": "value_at_risk", "params": {"window": 126}},
        {"type": "lstm", "params": {"sequence_length": 30, "epochs": 3}},
    ],
    "mews": {"enabled": False},  # set to True to retrain the full MEWS ensemble
})

ExperimentManager(config).run()
```

> **Dependencies:** the baselines utilise optional packages (`arch`, `torch`). Install them with `pip install -r requirements.txt` or remove the corresponding baseline entry if you want to skip them.

### Robustness Diagnostics

- **`src/robustness/`** bundles tooling to stress test the pipeline:
  - `SentimentBiasAuditor`: compares FinBERT and VADER distributions for polarity skew
  - `AdversarialNoiseTester`: injects Gaussian or uniform noise into tabular features
  - `DelaySimulator`: shifts and masks news sentiment columns to emulate reporting delays
  - `RobustnessEvaluator`: orchestrates perturbations, re-trains MEWS models, computes CEWS deltas, and writes JSON reports under `outputs/robustness/`

Run a full robustness sweep against `data/integrated_dataset.csv` with:

```bash
python robustness_eval.py data/integrated_dataset.csv
```

Add `--feature-groups path/to/feature_groups.json` to reuse curated feature lists or `--skip-bias` if the sentiment audit should be omitted. The evaluation produces a timestamped directory containing `robustness_report.json` with baseline metrics, per-perturbation deltas, and sentiment bias summaries.

### Hypothesis Testing Suite

- **`src/hypothesis/`** provides statistical comparisons tailored for research artifacts:
  - `paired_t_test` and `permutation_test` quantify gains from sentiment-aware models
  - `granger_causality` evaluates directional influence between sentiment and fundamental signals
  - `likelihood_ratio_test` supports nested model comparisons using log-likelihoods
  - `HypothesisReportBuilder` saves Markdown and HTML summaries to `outputs/hypothesis/`

Minimal example combining all diagnostics:

```python
from src.hypothesis import (
    HypothesisReportBuilder,
    granger_causality,
    likelihood_ratio_test,
    paired_t_test,
    permutation_test,
)

paired = [
    paired_t_test(sentiment_model_auc, baseline_auc, metric_name="AUC"),
    permutation_test(sentiment_model_cews, baseline_cews, metric_name="CEWS"),
]
granger = [granger_causality(sentiment_series=sentiment, fundamental_series=fundamental, max_lag=3)]
lr = [likelihood_ratio_test(null_model=null_result, alt_model=alt_result, model_name="Sentiment Feature")]

HypothesisReportBuilder().build_reports(
    paired_results=paired,
    granger_results=granger,
    lr_results=lr,
    metadata={"Dataset": "Integrated"},
)
```

The builder returns paths for both Markdown and HTML documents that can be cited in research notes or appended to the full MEWS report.

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

### Sample Visuals

| Risk Timeline | Sentiment Risk Timeline |
| --- | --- |
| ![Market sentiment timeline](outputs/sentiment_analysis.png) |

### Sample News Sentiment & Risk Snapshot

| Symbol | Date | SEC MD&A Sentiment | SEC Risk Sentiment | Combined Sentiment | Risk Probability |
| --- | --- | --- | --- | --- | --- |
| GOOGL | 2025-10-01 | 0.917 | 0.891 | 0.888 | 0.04 |
| MSFT | 2025-10-01 | 0.919 | 0.893 | 0.864 | 0.05 |
| NVDA | 2025-10-01 | — | — | 0.864 | 0.05 |

*NVDA currently has no recent SEC filing in the cached dataset, so only headline-driven sentiment is shown.*

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
- Research addendum (Markdown + HTML in `outputs/research/`)

### Research Artifacts
- Crisis-era benchmarks versus GARCH/VaR and LSTM baselines
- Likelihood-ratio hypothesis tests for sentiment and graph features
- Robustness diagnostics covering sentiment bias and adversarial noise

### Data Files
- `integrated_data.csv`: Combined features from all sources
- `risk_predictions.csv`: Model predictions with probabilities
- `sentiment_aggregated.csv`: Daily sentiment scores
- `news_sentiment_timeseries.csv`: Recent headline sentiment timeline used for README visuals
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

## 🧠 Research-Grade Enhancements

- **Regime-Adaptive Ensemble:** Learns volatility-aware weights across Random Forest, Logistic Regression, XGBoost, and SVM models.
- **Cross-Attention Fusion:** Optional transformer-based fusion layer that combines textual sentiment embeddings with tabular indicators via `CrossAttentionFusion`.
- **Benchmark Suite:** Compare MEWS against GARCH/VaR and LSTM baselines with crisis-period metrics including AUC, Brier score, and Precision@K.
- **Hypothesis Testing:** Quantify the lift from sentiment and graph features using likelihood-ratio tests and automated ablation studies.
- **Robustness & Bias Checks:** Diagnose sentiment skew and evaluate adversarial scenarios (noise or delayed news) with `SentimentBiasDetector` and `RobustnessStressTester`.
- **Research Reports:** Auto-generated Markdown/HTML summaries in `outputs/research/` ready for citation (see `docs/RESEARCH_GUIDE.md`).

## 🧬 Unique & Novel Contributions

- **Volatility-Regime Smarts:** A regime-adaptive ensemble continuously learns optimal weights as market volatility shifts, improving crisis detection.
- **Multimodal Attention Fusion:** Cross-attention bridges language models and tabular indicators so textual sentiment directly modulates numerical signals.
- **Publication-Ready Benchmarking:** Automated comparisons against GARCH/VaR and deep LSTM baselines yield crisis-specific diagnostics in a single run.
- **Integrated Causal Diagnostics:** Likelihood-ratio tests, graph feature ablations, and sentiment bias audits are bundled into the default pipeline to validate signal provenance.
- **Hands-Free Research Artifacts:** MEWS exports Markdown and HTML reports with methodology, figures, and robustness narratives ready for peer review.

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

