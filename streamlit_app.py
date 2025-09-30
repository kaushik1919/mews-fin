"""
Streamlit Frontend for Market Risk Early Warning System
Interactive dashboard with 4 ML models comparison
"""

import json
import os
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Import our system components
import sys

sys.path.append(".")
from datetime import datetime, timedelta

import requests


def _normalize_datetime_column(
    df: pd.DataFrame, column: str = "Date", dropna: bool = True
) -> pd.DataFrame:
    """Convert datetime-like column to timezone-naive pandas datetime."""

    if column in df.columns:
        as_str = df[column].astype(str)
        normalized = pd.to_datetime(as_str, errors="coerce", utc=True)
        df[column] = normalized.dt.tz_localize(None)
        if dropna:
            df = df.dropna(subset=[column])
    return df


def _format_dataframe_for_display(
    df: pd.DataFrame, date_columns: Optional[Tuple[str, ...]] = None
) -> pd.DataFrame:
    """Convert datetime-like columns to ISO strings for Streamlit display."""

    formatted = df.copy()

    if date_columns is None:
        inferred = []
        for column in formatted.columns:
            series = formatted[column]
            if pd.api.types.is_datetime64_any_dtype(series):
                inferred.append(column)
            elif (
                series.dtype == object
                and series.apply(
                    lambda val: isinstance(val, (pd.Timestamp, datetime))
                ).any()
            ):
                inferred.append(column)
        candidate_columns = tuple(inferred)
    else:
        candidate_columns = date_columns

    for column in candidate_columns:
        if column not in formatted.columns:
            continue

        series = formatted[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            formatted[column] = series.dt.strftime("%Y-%m-%d")
        else:
            formatted[column] = series.apply(
                lambda val: (
                    val.strftime("%Y-%m-%d")
                    if isinstance(val, (pd.Timestamp, datetime))
                    else val
                )
            )

    return formatted


def _render_dataframe(df: pd.DataFrame, **kwargs: Any) -> None:
    st.dataframe(_format_dataframe_for_display(df), **kwargs)


from src.backtester import RiskBacktester
from src.metrics import compute_cews_score
from src.xai_utils import (
    compute_global_shap_importance,
    compute_lime_explanation,
    compute_local_shap_explanation,
)

# Import enhanced features
try:
    from enhanced_integration import integrate_enhanced_risk_system

    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Market Risk Early Warning System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .gpu-status {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


def load_cached_data():
    """Load cached results if available"""
    # Try dataset with risk labels first
    cache_file_with_risk = "data/dataset_with_risk_labels.csv"
    cache_file = "data/integrated_dataset.csv"
    results_file = "data/model_results.json"

    # Load results
    results = None
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)

    # Load dataset (prefer one with risk labels)
    df = None
    if os.path.exists(cache_file_with_risk):
        df = pd.read_csv(cache_file_with_risk)
        df = _normalize_datetime_column(df)
        st.info("✅ Loaded dataset with pre-computed risk labels")
    elif os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        df = _normalize_datetime_column(df)
        # Generate risk labels if needed
        if "Risk_Label" not in df.columns and results:
            try:
                from src.ml_models import RiskPredictor

                predictor = RiskPredictor()
                X, y, feature_names = predictor.prepare_modeling_data(df)
                df["Risk_Label"] = y
                st.info("✅ Generated risk labels for timeline analysis")
            except Exception as e:
                st.warning(f"Could not generate risk labels: {e}")

    if df is not None and results is not None:
        return df, results
    return None, None


def fetch_gnews_sentiment(symbols, api_key, days=7):
    """Fetch news sentiment from GNews API"""
    if not api_key:
        return None

    all_news = []

    for symbol in symbols:
        try:
            # GNews API endpoint - using broader search without date restrictions for free plan
            url = "https://gnews.io/api/v4/search"
            params = {
                "q": f"{symbol} stock",  # Simplified query
                "token": api_key,
                "lang": "en",
                "country": "us",
                "max": 15,  # Get more articles per symbol
                "sortby": "publishedAt",  # Most recent first
            }

            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()

                # Handle API messages
                if "information" in data:
                    st.info(
                        "📰 Using GNews free plan - articles may have 12-hour delay"
                    )

                articles = data.get("articles", [])
                st.success(f"Found {len(articles)} articles for {symbol}")

                for article in articles:
                    all_news.append(
                        {
                            "symbol": symbol,
                            "title": article.get("title", ""),
                            "description": article.get("description", ""),
                            "content": article.get("content", ""),
                            "published_date": article.get("publishedAt", ""),
                            "source": article.get("source", {}).get("name", ""),
                            "url": article.get("url", ""),
                        }
                    )
            else:
                st.error(
                    f"API Error for {symbol}: {response.status_code} - {response.text[:200]}"
                )

        except Exception as e:
            st.error(f"Error fetching news for {symbol}: {str(e)}")

    return pd.DataFrame(all_news) if all_news else None


def analyze_news_sentiment(news_df):
    """Analyze sentiment of news articles with detailed progress"""
    if news_df is None or news_df.empty:
        return None

    try:
        from src.sentiment_analyzer import SentimentAnalyzer

        analyzer = SentimentAnalyzer()

        # Combine title, description, and content for better analysis
        news_df["full_text"] = (
            news_df["title"].fillna("")
            + " "
            + news_df["description"].fillna("")
            + " "
            + news_df["content"].fillna("")
        )

        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        sentiments = []
        total_articles = len(news_df)

        for i, text in enumerate(news_df["full_text"]):
            # Update progress
            progress = (i + 1) / total_articles
            progress_bar.progress(progress)
            status_text.text(f"Analyzing sentiment: {i+1}/{total_articles} articles")

            if text.strip() and len(text) > 10:  # Only analyze meaningful text
                try:
                    # Use VADER directly for quick sentiment analysis
                    if analyzer.vader_analyzer:
                        sentiment = analyzer.vader_analyzer.polarity_scores(text)
                        sentiments.append(sentiment)
                    else:
                        sentiments.append({"compound": 0, "pos": 0, "neu": 1, "neg": 0})
                except:
                    # Fallback to neutral if analysis fails
                    sentiments.append({"compound": 0, "pos": 0, "neu": 1, "neg": 0})
            else:
                sentiments.append({"compound": 0, "pos": 0, "neu": 1, "neg": 0})

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Add sentiment scores to dataframe
        for i, sent in enumerate(sentiments):
            news_df.loc[i, "sentiment_compound"] = sent["compound"]
            news_df.loc[i, "sentiment_positive"] = sent["pos"]
            news_df.loc[i, "sentiment_negative"] = sent["neg"]
            news_df.loc[i, "sentiment_neutral"] = sent["neu"]

            # Add human-readable sentiment labels
            compound = sent["compound"]
            if compound >= 0.05:
                news_df.loc[i, "sentiment_label"] = "Positive 📈"
            elif compound <= -0.05:
                news_df.loc[i, "sentiment_label"] = "Negative 📉"
            else:
                news_df.loc[i, "sentiment_label"] = "Neutral ➡️"

        st.success(f"✅ Sentiment analysis completed for {len(news_df)} articles")
        return news_df

    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")
        return news_df


def create_sentiment_risk_chart(news_df, stock_data):
    """Create combined sentiment and risk timeline"""
    if news_df is None or news_df.empty:
        return None

    try:
        # Group news sentiment by symbol and date
        news_df = _normalize_datetime_column(news_df, "published_date")
        news_df["date"] = news_df["published_date"].dt.date

        daily_sentiment = (
            news_df.groupby(["symbol", "date"])
            .agg(
                {
                    "sentiment_compound": "mean",
                    "sentiment_positive": "mean",
                    "sentiment_negative": "mean",
                }
            )
            .reset_index()
        )

        # Create the combined chart
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("News Sentiment Timeline", "Risk Level Timeline"),
            vertical_spacing=0.1,
        )

        # Add sentiment timeline
        for symbol in daily_sentiment["symbol"].unique():
            symbol_data = daily_sentiment[daily_sentiment["symbol"] == symbol]

            fig.add_trace(
                go.Scatter(
                    x=symbol_data["date"],
                    y=symbol_data["sentiment_compound"],
                    mode="lines+markers",
                    name=f"{symbol} Sentiment",
                    line=dict(width=2),
                ),
                row=1,
                col=1,
            )

        # Add risk timeline if available
        if stock_data is not None and "Risk_Label" in stock_data.columns:
            if "Date" in stock_data.columns:
                stock_data = _normalize_datetime_column(stock_data, "Date")
                daily_risk = (
                    stock_data.groupby("Date")["Risk_Label"].mean().reset_index()
                )

                fig.add_trace(
                    go.Scatter(
                        x=daily_risk["Date"],
                        y=daily_risk["Risk_Label"],
                        mode="lines+markers",
                        name="Market Risk",
                        line=dict(color="red", width=2),
                    ),
                    row=2,
                    col=1,
                )

        fig.update_layout(
            title="Combined Sentiment & Risk Analysis", height=600, showlegend=True
        )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Sentiment Score", row=1, col=1)
        fig.update_yaxes(title_text="Risk Level", row=2, col=1)

        return fig

    except Exception as e:
        st.error(f"Error creating sentiment chart: {str(e)}")
        return None


def create_model_comparison_chart(results):
    """Create interactive model comparison chart"""
    if not results:
        return None

    models = []
    accuracies = []
    auc_scores = []

    for model_name, metrics in results.items():
        if model_name not in ["test_data", "ensemble"]:
            models.append(model_name.replace("_", " ").title())
            accuracies.append(metrics.get("test_accuracy", 0))
            auc_scores.append(metrics.get("auc_score", 0))

    # Add ensemble
    if "ensemble" in results:
        models.append("Ensemble")
        accuracies.append(results["ensemble"].get("test_accuracy", 0))
        auc_scores.append(results["ensemble"].get("auc_score", 0))

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Test Accuracy", "AUC Score"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]],
    )

    # Accuracy bars
    fig.add_trace(
        go.Bar(
            x=models,
            y=accuracies,
            name="Accuracy",
            marker_color="lightblue",
            text=[f"{acc:.3f}" for acc in accuracies],
            textposition="auto",
        ),
        row=1,
        col=1,
    )

    # AUC bars
    fig.add_trace(
        go.Bar(
            x=models,
            y=auc_scores,
            name="AUC Score",
            marker_color="lightcoral",
            text=[f"{auc:.3f}" for auc in auc_scores],
            textposition="auto",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="ML Model Performance Comparison", showlegend=False, height=400
    )

    return fig


def create_feature_importance_chart(results):
    """Create user-friendly feature importance visualization"""
    if not results or "random_forest" not in results:
        return None

    rf_results = results["random_forest"]
    feature_importance = rf_results.get("feature_importance")
    if not feature_importance:
        return None

    sorted_features = sorted(
        feature_importance.items(), key=lambda item: item[1], reverse=True
    )[:15]

    feature_names = [name.replace("_", " ").title() for name, _ in sorted_features]
    values = [value for _, value in sorted_features]

    fig = go.Figure(
        data=[
            go.Bar(
                x=feature_names,
                y=values,
                marker=dict(color="crimson"),
                text=[f"{value:.3f}" for value in values],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="Top Factors Influencing Market Risk",
        xaxis_tickangle=-45,
        height=500,
    )

    return fig


@st.cache_resource(show_spinner=False)
def load_latest_model(model_name: str = "random_forest") -> Optional[object]:
    """Load the most recent trained model artifact from the models directory."""

    models_root = Path("models")
    if not models_root.exists():
        return None

    candidates = sorted(models_root.glob("models_*"))
    if not candidates:
        return None

    latest_dir = max(candidates, key=lambda path: path.name)
    model_path = latest_dir / f"{model_name}_model.pkl"
    if not model_path.exists():
        return None

    with open(model_path, "rb") as handle:
        return pickle.load(handle)


@st.cache_data(show_spinner=False)
def prepare_feature_matrix(
    df: pd.DataFrame,
    target_column: str = "Risk_Label",
    feature_order: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Prepare numeric feature matrix aligned to the trained model's features."""

    numeric_features = df.select_dtypes(include=[np.number]).copy()
    if target_column in numeric_features.columns:
        numeric_features = numeric_features.drop(columns=[target_column])

    numeric_features = numeric_features.fillna(0)

    if feature_order is None:
        return numeric_features

    ordered_columns = list(feature_order)
    ordered = numeric_features.reindex(columns=ordered_columns, fill_value=0.0)
    return ordered


def get_test_predictions(
    results: Dict[str, Any],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Extract ground-truth labels and ensemble probabilities from stored results."""

    test_data = results.get("test_data")
    if not test_data:
        return None

    y_true = np.array(test_data.get("y_test", []))
    if y_true.size == 0:
        return None

    for key in ["ensemble_probabilities", "rf_probabilities", "lr_probabilities"]:
        if key in test_data:
            y_prob = np.array(test_data[key])
            break
    else:
        return None

    if y_prob.size != y_true.size:
        return None

    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


def create_confusion_matrix_plot(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Optional[go.Figure]:
    """Create an annotated confusion matrix using Plotly."""

    if y_true.size == 0 or y_pred.size == 0:
        return None

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    labels = [["TN", "FP"], ["FN", "TP"]]
    heatmap = ff.create_annotated_heatmap(
        z=cm,
        x=["Predicted 0", "Predicted 1"],
        y=["Actual 0", "Actual 1"],
        annotation_text=labels,
        colorscale="Blues",
        showscale=True,
    )
    heatmap.update_layout(title="Confusion Matrix", height=400)
    return heatmap


@st.cache_data(show_spinner=False)
def run_cached_backtest(
    df: pd.DataFrame, predictions_df: Optional[pd.DataFrame]
) -> Optional[Dict[str, Any]]:
    """Run comprehensive backtest with caching to avoid recomputation."""

    if df.empty:
        return None

    try:
        backtester = RiskBacktester()
        return backtester.run_comprehensive_backtest(df, predictions_df)
    except Exception as exc:  # pragma: no cover - backtester optional
        st.warning(f"Backtesting unavailable: {exc}")
        return None


def build_cews_input(df: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, str]]:
    """Prepare dataframe and probability column for CEWS computation."""

    if df.empty:
        return None

    working_df = df.copy()
    working_df = _normalize_datetime_column(working_df, "Date")
    if "Date" not in working_df.columns or "Risk_Label" not in working_df.columns:
        return None

    probability_col = None
    if "Risk_Probability" in working_df.columns:
        probability_col = "Risk_Probability"
    elif "Risk_Score" in working_df.columns:
        probability_col = "Risk_Score"
    else:
        working_df["Risk_Probability"] = (
            working_df["Risk_Label"].rolling(window=3, min_periods=1).mean()
        )
        probability_col = "Risk_Probability"

    cews_df = working_df.dropna(subset=["Date", "Risk_Label", probability_col])
    return cews_df, probability_col


def explain_features():
    """Provide user-friendly explanations of key features"""
    explanations = {
        "RSI (Price Momentum)": "📈 Shows if a stock is overbought (>70) or oversold (<30). High values often signal price corrections.",
        "MACD (Trend Indicator)": "📊 Measures trend changes. When MACD crosses above signal line, it suggests upward momentum.",
        "Bollinger Bands": "📏 Shows price volatility. When price hits upper band, stock might be overvalued.",
        "Moving Average": "📉 Smoothed price over time. When current price is below moving average, it may signal downtrend.",
        "Trading Volume": "📦 Number of shares traded. High volume with price drops often signals strong selling pressure.",
        "Price Volatility": "⚡ How much price swings up and down. High volatility = higher risk.",
        "News Sentiment": "📰 Positive/negative tone in news articles. Negative sentiment often precedes price drops.",
        "Market Correlation": "🔗 How closely stock moves with overall market. High correlation = more affected by market crashes.",
        "VIX (Fear Index)": "😰 Market's expectation of volatility. VIX > 30 suggests high market fear.",
    }
    return explanations


def create_risk_timeline(df):
    """Create enhanced risk timeline visualization"""
    if df is None or df.empty:
        return None

    try:
        # Check for required columns
        if "Date" not in df.columns or "Risk_Label" not in df.columns:
            st.warning("Risk timeline requires 'Date' and 'Risk_Label' columns")
            return None

        # Prepare data
        timeline_data = df.copy()

        # Handle different date formats
        try:
            timeline_data = _normalize_datetime_column(timeline_data, "Date")
        except Exception:
            st.warning("Could not parse Date column for risk timeline")
            return None

        # Group by date and calculate risk metrics
        daily_risk = (
            timeline_data.groupby("Date")
            .agg({"Risk_Label": ["mean", "std", "count"], "Symbol": "nunique"})
            .round(3)
        )

        daily_risk.columns = [
            "Risk_Level",
            "Risk_Volatility",
            "Sample_Count",
            "Symbols_Count",
        ]
        daily_risk = daily_risk.reset_index()

        # Create the enhanced timeline
        fig = go.Figure()

        # Main risk line
        fig.add_trace(
            go.Scatter(
                x=daily_risk["Date"],
                y=daily_risk["Risk_Level"],
                mode="lines+markers",
                name="Risk Level",
                line=dict(color="red", width=3),
                marker=dict(size=6),
                hovertemplate="<b>Date:</b> %{x}<br>"
                + "<b>Risk Level:</b> %{y:.3f}<br>"
                + "<b>Samples:</b> %{customdata[0]}<br>"
                + "<b>Symbols:</b> %{customdata[1]}<extra></extra>",
                customdata=daily_risk[["Sample_Count", "Symbols_Count"]],
            )
        )

        # Add risk zones
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="orange",
            annotation_text="Medium Risk Threshold",
        )
        fig.add_hline(
            y=0.7,
            line_dash="dash",
            line_color="red",
            annotation_text="High Risk Threshold",
        )

        # Color background zones
        fig.add_hrect(
            y0=0,
            y1=0.3,
            fillcolor="green",
            opacity=0.1,
            annotation_text="Low Risk Zone",
            annotation_position="top left",
        )
        fig.add_hrect(
            y0=0.3,
            y1=0.7,
            fillcolor="yellow",
            opacity=0.1,
            annotation_text="Medium Risk Zone",
            annotation_position="top left",
        )
        fig.add_hrect(
            y0=0.7,
            y1=1,
            fillcolor="red",
            opacity=0.1,
            annotation_text="High Risk Zone",
            annotation_position="top left",
        )

        fig.update_layout(
            title="📊 Market Risk Timeline - Easy to Read",
            xaxis_title="Date",
            yaxis_title="Risk Level (0=Safe, 1=Very Risky)",
            height=500,
            hovermode="x unified",
            showlegend=True,
        )

        return fig

    except Exception as e:
        st.error(f"Error creating risk timeline: {str(e)}")
        return None


def main():
    """Main Streamlit app"""

    # Header
    st.markdown(
        '<h1 class="main-header">📊 Market Risk Early Warning System</h1>',
        unsafe_allow_html=True,
    )

    # System Status
    st.info("📊 Market Risk Analysis System - Real-time sentiment integration")

    # Sidebar
    st.sidebar.title("Configuration")

    # Load cached data
    df, results = load_cached_data()

    if df is None or results is None:
        st.sidebar.warning("No cached data found. Please run the pipeline first:")
        st.sidebar.code("python main.py --full-pipeline")

        st.error(
            "❌ No data available. Please run the analysis first using the command line tool."
        )
        st.info("Run: `python main.py --full-pipeline` to generate data")
        return

    # Data summary
    st.sidebar.success(f"✅ Data loaded: {len(df)} samples")
    st.sidebar.info(f"Features: {len(df.columns)}")
    st.sidebar.info(
        f"Symbols: {df['Symbol'].nunique() if 'Symbol' in df.columns else 'N/A'}"
    )

    # Prepare derived datasets for advanced visualizations
    y_true_pred = get_test_predictions(results)
    cews_bundle = build_cews_input(df)
    cews_df: Optional[pd.DataFrame] = None
    probability_column: Optional[str] = None
    predictions_for_backtest: Optional[pd.DataFrame] = None

    if cews_bundle:
        cews_df, probability_column = cews_bundle
        predictions_for_backtest = cews_df.copy()
        if "Symbol" not in predictions_for_backtest.columns:
            predictions_for_backtest["Symbol"] = "MEWS"
        predictions_for_backtest = predictions_for_backtest.rename(
            columns={probability_column: "Risk_Probability"}
        )

    (
        tab_model_results,
        tab_explainability,
        tab_multimodal,
        tab_feature_analysis,
        tab_risk_timeline,
        tab_sentiment,
        tab_data_explorer,
        tab_model_details,
    ) = st.tabs(
        [
            "📊 Model Results",
            "🧠 Explainability",
            "🧬 Multimodal + CEWS",
            "📈 Feature Analysis",
            "📊 Risk Timeline",
            "📰 Sentiment Analysis",
            "🔍 Data Explorer",
            "⚙️ Model Details",
        ]
    )

    with tab_model_results:
        st.subheader("Comprehensive Model Results")

        summary_cols = st.columns(4)
        model_count = 0
        total_accuracy = 0.0
        best_model = "N/A"
        best_auc = 0.0

        for model_name, metrics in results.items():
            if not isinstance(metrics, dict) or model_name == "test_data":
                continue

            model_count += 1
            accuracy = float(metrics.get("test_accuracy", 0.0))
            auc_score = float(metrics.get("auc_score", 0.0))
            total_accuracy += accuracy

            if model_name != "ensemble" and auc_score > best_auc:
                best_auc = auc_score
                best_model = model_name.replace("_", " ").title()

        summary_cols[0].metric("Models Trained", model_count)
        summary_cols[1].metric(
            "Avg Accuracy", f"{total_accuracy / max(model_count, 1):.3f}"
        )
        summary_cols[2].metric("Best Model", best_model)
        summary_cols[3].metric("Best AUC", f"{best_auc:.3f}")

        if y_true_pred:
            from sklearn.metrics import precision_score, recall_score, roc_auc_score

            y_true, y_pred, y_prob = y_true_pred
            metric_cols = st.columns(3)
            metric_cols[0].metric(
                "Ensemble AUC", f"{roc_auc_score(y_true, y_prob):.3f}"
            )
            metric_cols[1].metric(
                "Precision", f"{precision_score(y_true, y_pred, zero_division=0):.3f}"
            )
            metric_cols[2].metric(
                "Recall", f"{recall_score(y_true, y_pred, zero_division=0):.3f}"
            )

            confusion_fig = create_confusion_matrix_plot(y_true, y_pred)
            if confusion_fig:
                st.plotly_chart(confusion_fig, use_container_width=True)
        else:
            st.info(
                "Test prediction data unavailable. Run the training pipeline to populate evaluation metrics."
            )

        comparison_fig = create_model_comparison_chart(results)
        if comparison_fig:
            st.plotly_chart(
                comparison_fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

        st.subheader("Detailed Model Results")
        rows = []
        for model_name, metrics in results.items():
            if not isinstance(metrics, dict) or model_name == "test_data":
                continue
            rows.append(
                {
                    "Model": model_name.replace("_", " ").title(),
                    "Accuracy": f"{metrics.get('test_accuracy', 0):.4f}",
                    "AUC Score": f"{metrics.get('auc_score', 0):.4f}",
                    "Models Used": (
                        ", ".join(metrics.get("models_used", []))
                        if isinstance(metrics.get("models_used"), list)
                        else "-"
                    ),
                }
            )

        if rows:
            _render_dataframe(pd.DataFrame(rows), use_container_width=True)

        if predictions_for_backtest is not None:
            backtest_results = run_cached_backtest(df, predictions_for_backtest)
            event_analysis = (
                backtest_results.get("event_analysis") if backtest_results else None
            )

            if event_analysis:
                st.subheader("Historical Stress Tests")
                event_rows = []
                for event_key, analysis in event_analysis.items():
                    if analysis.get("status") == "no_data":
                        continue

                    risk_signals = analysis.get("risk_signals", {})
                    prediction_perf = analysis.get("prediction_performance", {})
                    avg_prob = prediction_perf.get("avg_risk_probability")
                    event_rows.append(
                        {
                            "Event": analysis.get(
                                "description", event_key.replace("_", " ").title()
                            ),
                            "Period": analysis.get("period", "N/A"),
                            "Risk Rate": f"{risk_signals.get('risk_rate', 0):.2%}",
                            "Early Warning": f"{analysis.get('early_warning_score', 0.0):+.2f}",
                            "Avg Predicted Risk": (
                                "N/A" if avg_prob is None else f"{avg_prob:.2f}"
                            ),
                        }
                    )

                if event_rows:
                    _render_dataframe(
                        pd.DataFrame(event_rows), use_container_width=True
                    )
                    st.caption(
                        "Positive early warning scores indicate the system raised alerts before the event window."
                    )
                else:
                    st.info(
                        "No historical market event windows overlapped with the loaded dataset."
                    )
        else:
            st.info(
                "CEWS inputs not available yet—generate probability forecasts to unlock backtesting analytics."
            )

    with tab_explainability:
        st.subheader("Explainability Studio (SHAP & LIME)")
        st.info(
            "This area breaks down WHY the model thinks conditions are risky. "
            "The big blue chart summarizes the top ingredients driving the model across your sample, "
            "while the two smaller bar charts explain a single day for a single ticker—showing which signals pushed the call toward risk (blue/green) or toward stability (red). "
            "Use the sample-size slider to choose how much history to summarize: larger samples smooth noise, smaller ones highlight the latest moves."
        )
        model = load_latest_model()

        feature_order: Optional[Iterable[str]] = None
        if model is not None:
            feature_names_attr = getattr(model, "feature_names_in_", None)
            if feature_names_attr is not None:
                feature_order = list(feature_names_attr)
            elif isinstance(results.get("random_forest"), dict):
                importance = results["random_forest"].get("feature_importance")
                if isinstance(importance, dict):
                    feature_order = list(importance.keys())

        feature_matrix = prepare_feature_matrix(df, feature_order=feature_order)

        if model is None:
            st.warning(
                "No trained model artifacts found. Run the training pipeline to enable explainability insights."
            )
        elif feature_matrix.empty:
            st.warning("Feature matrix is empty; cannot compute explanations.")
        else:
            sample_max = min(len(feature_matrix), 1000)
            sample_min = min(100, sample_max)
            sample_default = min(500, sample_max)
            sample_size = st.slider(
                "Sample size for global SHAP",
                sample_min,
                sample_max,
                sample_default,
                step=50,
            )

            shap_input = (
                feature_matrix.sample(sample_size, random_state=42)
                if len(feature_matrix) > sample_size
                else feature_matrix
            )

            with st.spinner("Computing global SHAP importance..."):
                shap_global = compute_global_shap_importance(model, shap_input)

            if shap_global is not None and not shap_global.empty:
                top_global = shap_global.head(20)
                global_fig = go.Figure(
                    go.Bar(
                        x=top_global["importance"][::-1],
                        y=top_global["feature"][::-1],
                        orientation="h",
                        marker=dict(color="#2563eb"),
                    )
                )
                global_fig.update_layout(
                    title="Global Feature Importance (Mean |SHAP|)",
                    height=500,
                    margin=dict(l=150, r=30, t=60, b=40),
                )
                st.plotly_chart(global_fig, use_container_width=True)
                st.markdown(
                    """
                    **How to read it:** Each bar shows the average absolute SHAP impact for a feature across the sampled rows. Longer bars mean the feature consistently nudged the model's risk score up or down. A flat bar implies the model rarely relied on that input. Look for clusters of related features (e.g., different volatility measures) to understand combined pressure on the prediction.

                    • Values are always positive because we take the average magnitude of the push.  
                    • If two features are close in height, they contributed nearly the same amount of influence.  
                    • Use this chart to spot which signals dominate the model so you can monitor or stress-test them directly.
                    """
                )
            else:
                st.info(
                    "SHAP library not installed; run `pip install shap` to unlock global explanations."
                )

            st.markdown("### Local Explanations")
            selected_position: Optional[int] = None

            if {"Symbol", "Date"}.issubset(df.columns):
                selection_df = df[["Symbol", "Date"]].copy()
                selection_df["row_position"] = np.arange(len(selection_df))
                selection_df = _normalize_datetime_column(selection_df, "Date")

                if not selection_df.empty:
                    symbol_choice = st.selectbox(
                        "Ticker", sorted(selection_df["Symbol"].dropna().unique())
                    )
                    symbol_slice = selection_df[selection_df["Symbol"] == symbol_choice]
                    date_options = (
                        symbol_slice["Date"].dt.date.unique()
                        if not symbol_slice.empty
                        else []
                    )
                    if len(date_options) > 0:
                        date_choice = st.selectbox(
                            "Date", sorted(date_options, reverse=True)
                        )
                        match_rows = symbol_slice[
                            symbol_slice["Date"].dt.date == date_choice
                        ]
                        if not match_rows.empty:
                            selected_position = int(match_rows["row_position"].iloc[0])

            if selected_position is None:
                selected_position = 0

            selected_position = int(
                np.clip(selected_position, 0, max(len(feature_matrix) - 1, 0))
            )

            with st.spinner("Computing local SHAP and LIME explanations..."):
                shap_local = compute_local_shap_explanation(
                    model, feature_matrix, selected_position
                )
                lime_explanation = compute_lime_explanation(
                    model,
                    feature_matrix,
                    selected_position,
                    class_names=np.array(["Stable", "Risk"]),
                )

            if shap_local is not None and not shap_local.empty:
                local_display = shap_local.head(20).sort_values("abs_shap")
                local_colors = [
                    "#16a34a" if val >= 0 else "#dc2626"
                    for val in local_display["shap_value"]
                ]
                local_fig = go.Figure(
                    go.Bar(
                        x=local_display["shap_value"],
                        y=local_display["feature"],
                        orientation="h",
                        marker_color=local_colors,
                    )
                )
                local_fig.update_layout(
                    title="Local SHAP Contributions",
                    height=500,
                    margin=dict(l=150, r=30, t=60, b=40),
                )
                st.plotly_chart(local_fig, use_container_width=True)
                st.markdown(
                    """
                    **Understanding the bars:** Positive (green) bars pushed the prediction toward **Risk**, while negative (red) bars pulled it toward **Stable**. The bar length equals that feature's SHAP value for the selected date/ticker. Compare these against the global chart to see whether today's drivers match the long-term leaders.

                    • SHAP values add up to the model's risk score once you include the baseline probability.  
                    • Large positive + large negative bars can offset each other; the mix tells you whether the day was borderline or clearly risky.  
                    • Hover each bar to see the exact contribution in probability points.
                    """
                )
            else:
                st.info("Local SHAP explanations unavailable for the selected sample.")

            if lime_explanation is not None and not lime_explanation.empty:
                lime_display = lime_explanation.sort_values("weight")
                lime_colors = [
                    "#16a34a" if val >= 0 else "#dc2626"
                    for val in lime_display["weight"]
                ]
                lime_fig = go.Figure(
                    go.Bar(
                        x=lime_display["weight"],
                        y=lime_display["feature"],
                        orientation="h",
                        marker_color=lime_colors,
                    )
                )
                lime_fig.update_layout(
                    title="LIME Feature Weights",
                    height=500,
                    margin=dict(l=150, r=30, t=60, b=40),
                )
                st.plotly_chart(lime_fig, use_container_width=True)
                st.caption(
                    "LIME perturbs the original row to learn a tiny linear model around it. Positive weights (green) argue for the risk class, negative weights (red) argue for the stable class. Compare them with the SHAP bars: if both methods agree on the top signals, the explanation is more trustworthy."
                )
            else:
                st.info(
                    "LIME explanation unavailable—install the `lime` package to enable this view."
                )

    with tab_multimodal:
        st.subheader("Multimodal Fusion & CEWS Insights")
        st.info(
            "This view highlights fused tabular, news, and graph features alongside the Crisis Early Warning Score (CEWS)."
        )
        st.markdown(
            """
            **In plain terms:** we blend classic market stats, news mood, and network-style signals into a single early-warning dashboard.

            • **CEWS Score (0→1)** – your headline risk gauge; values above ~0.7 mean “high alert.”  
            • **Early Detection Reward** – higher is better; shows how often we caught danger before it hit.  
            • **False Alarm Penalty** – lower is better; tells you if we’re crying wolf.  
            • Watch the three lines on the timeline: when the blue CEWS line climbs and stays above the green reward line while the red penalty line remains low, the system is confident about rising risk.
            """
        )

        def _preview_columns(title: str, candidates: list[str]) -> None:
            subset = [col for col in candidates if col in df.columns]
            if subset:
                st.markdown(f"**{title}**")
                _render_dataframe(df[subset].head(5), use_container_width=True)
            else:
                st.markdown(f"**{title}:** _No features detected in dataset._")

        tabular_keywords = [
            "return",
            "volatility",
            "volume",
            "close",
            "open",
            "high",
            "low",
            "pe",
            "beta",
        ]
        news_keywords = ["sentiment", "news", "headline", "embedding"]
        graph_keywords = ["centrality", "pagerank", "graph", "community"]

        tabular_cols = [
            col
            for col in df.columns
            if any(key in col.lower() for key in tabular_keywords)
        ][:10]
        news_cols = [
            col
            for col in df.columns
            if any(key in col.lower() for key in news_keywords)
        ][:10]
        graph_cols = [
            col
            for col in df.columns
            if any(key in col.lower() for key in graph_keywords)
        ][:10]

        _preview_columns("Tabular Signals", tabular_cols)
        _preview_columns("News & Sentiment Signals", news_cols)
        _preview_columns("Graph Connectivity Signals", graph_cols)

        if cews_df is not None and probability_column is not None:
            cews_for_calc = _normalize_datetime_column(cews_df.copy(), "Date")
            if "Symbol" not in cews_for_calc.columns:
                cews_for_calc["Symbol"] = "MEWS"

            cews_ready = cews_for_calc.rename(
                columns={probability_column: "Risk_Probability"}
            )

            cews_result = compute_cews_score(
                cews_ready,
                probability_col="Risk_Probability",
                label_col="Risk_Label",
                date_col="Date",
                symbol_col="Symbol",
            )

            metric_cols = st.columns(3)
            metric_cols[0].metric("CEWS Score", f"{cews_result.score:.3f}")
            metric_cols[1].metric(
                "Early Detection Reward",
                f"{cews_result.early_detection_reward:.3f}",
            )
            metric_cols[2].metric(
                "False Alarm Penalty",
                f"{cews_result.false_alarm_penalty:.3f}",
            )

            timeline = cews_result.timeline.copy()
            timeline_fig = go.Figure()
            timeline_fig.add_trace(
                go.Scatter(
                    x=timeline["Date"],
                    y=timeline["cews"],
                    mode="lines",
                    name="CEWS",
                    line=dict(color="#2563eb", width=3),
                )
            )
            timeline_fig.add_trace(
                go.Scatter(
                    x=timeline["Date"],
                    y=timeline["early_reward"],
                    mode="lines",
                    name="Early Detection Reward",
                    line=dict(color="#16a34a", dash="dash", width=2),
                )
            )
            timeline_fig.add_trace(
                go.Scatter(
                    x=timeline["Date"],
                    y=timeline["false_alarm_penalty"],
                    mode="lines",
                    name="False Alarm Penalty",
                    line=dict(color="#dc2626", dash="dot", width=2),
                )
            )

            timeline_fig.update_layout(
                title="Crisis Early Warning Score Timeline",
                xaxis_title="Date",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1]),
                hovermode="x unified",
                template="plotly_white",
                height=500,
                legend=dict(orientation="h", y=1.08, x=0),
            )
            st.plotly_chart(timeline_fig, use_container_width=True)

            with st.expander("CEWS Metadata"):
                st.json(cews_result.metadata)
        else:
            st.info(
                "CEWS visualizations require probability forecasts. Train models or run the pipeline to generate these inputs."
            )

    with tab_feature_analysis:
        st.subheader("📈 What Drives Market Risk? (Plain English)")

        st.info(
            "💡 This shows which market factors are most important for predicting risk. Think of it like a recipe - these are the main ingredients that create market risk!"
        )

        # Feature importance chart
        importance_fig = create_feature_importance_chart(results)
        if importance_fig:
            config = {"displayModeBar": False}
            st.plotly_chart(importance_fig, use_container_width=True, config=config)

            # Add explanations
            st.subheader("🧠 What Do These Factors Mean?")
            explanations = explain_features()

            # Show explanations in an expandable format
            with st.expander("📚 Click to understand what each factor means"):
                for factor, explanation in explanations.items():
                    st.write(f"**{factor}**: {explanation}")

            # Key insights
            st.subheader("🎯 Key Insights")
            col1, col2 = st.columns(2)

            with col1:
                st.success("✅ **What This Tells Us:**")
                st.write("• Higher bars = more important for risk prediction")
                st.write("• Red bars = most critical factors to watch")
                st.write("• These factors work together to signal risk")

            with col2:
                st.warning("⚠️ **For Investors:**")
                st.write("• Watch the top 3-5 factors closely")
                st.write("• When multiple factors align, risk increases")
                st.write("• Technical indicators often lead fundamental news")
        else:
            st.warning(
                "Feature importance data not available - please run the models first"
            )

        # Feature correlation heatmap
        if len(df.select_dtypes(include=[np.number]).columns) > 1:
            st.subheader("🔗 How Market Factors Relate to Each Other")
            st.info(
                "💡 This heatmap shows which factors move together. Dark red = they move in the same direction, dark blue = they move opposite to each other."
            )

            numeric_cols = df.select_dtypes(include=[np.number]).columns[
                :15
            ]  # Limit for performance
            corr_matrix = df[numeric_cols].corr()

            fig = px.imshow(
                corr_matrix,
                title="Market Factor Relationships (Correlation Matrix)",
                color_continuous_scale="RdBu_r",
                labels=dict(color="Correlation"),
            )

            fig.update_layout(title="🔗 Market Factor Relationships", height=600)

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("❓ How to Read This Chart"):
                st.write("• **Dark Red (+1.0)**: Factors move perfectly together")
                st.write("• **White (0.0)**: No relationship between factors")
                st.write("• **Dark Blue (-1.0)**: Factors move in opposite directions")
                st.write("• **Light colors**: Weak relationships")
                st.write(
                    "• **Tip**: Look for clusters of red or blue to find factor groups"
                )

    with tab_risk_timeline:
        st.subheader("📊 Market Risk Timeline - Your Risk Radar")

        st.info(
            "💡 **What am I looking at?** This shows how risky the market was on different days. Think of it like a weather forecast - but for your investments!"
        )

        # Risk timeline
        timeline_fig = create_risk_timeline(df)
        if timeline_fig:
            # Configure plotly for better display
            config = {"displayModeBar": False}
            st.plotly_chart(timeline_fig, use_container_width=True, config=config)

            # Add explanations
            with st.expander("🧠 How to Read This Chart"):
                st.markdown(
                    """
                **🟢 Green Zone (0.0-0.3)**: **Safe to Invest**
                - Low risk period
                - Good time for conservative investors
                - Market conditions are stable
                
                **🟡 Yellow Zone (0.3-0.7)**: **Be Cautious**
                - Medium risk period  
                - Watch the market closely
                - Consider taking some profits
                
                **🔴 Red Zone (0.7-1.0)**: **High Alert!**
                - High risk period
                - Consider reducing positions
                - Market may be volatile or declining
                
                **📈 The Line Shows**: Daily risk level based on our AI models
                **🎯 Thresholds**: Dotted lines show when risk changes from low→medium→high
                """
                )
        else:
            st.error(
                "❌ Could not create risk timeline. Please ensure the data has been processed correctly."
            )

        # Enhanced Risk Timeline Option
        if ENHANCED_FEATURES_AVAILABLE:
            st.markdown("---")
            st.subheader("🚀 Enhanced Risk Timeline (New!)")

            if st.button("🎯 Generate Enhanced Risk Analysis", type="primary"):
                with st.spinner("🔄 Creating enhanced risk features and timeline..."):
                    try:
                        # Get selected symbols from sidebar
                        available_symbols = (
                            df["Symbol"].unique() if "Symbol" in df.columns else []
                        )
                        selected_symbols = st.sidebar.multiselect(
                            "Select symbols for enhanced analysis:",
                            available_symbols,
                            default=(
                                available_symbols[:3]
                                if len(available_symbols) >= 3
                                else available_symbols
                            ),
                        )

                        if selected_symbols:
                            # Run enhanced analysis
                            enhanced_results = integrate_enhanced_risk_system(
                                df, selected_symbols
                            )

                            if "enhanced_dataset" in enhanced_results:
                                st.success(
                                    f"✅ Enhanced analysis complete! Added {enhanced_results.get('feature_count', 0)} new features"
                                )

                                # Show enhanced timeline link
                                if enhanced_results.get("timeline_path"):
                                    st.markdown(
                                        f"📊 **Enhanced Interactive Timeline**: [View Enhanced Dashboard]({enhanced_results['timeline_path']})"
                                    )

                                # Show summary dashboard link
                                if enhanced_results.get("summary_dashboard_path"):
                                    st.markdown(
                                        f"📋 **Risk Summary Dashboard**: [View Summary]({enhanced_results['summary_dashboard_path']})"
                                    )

                                # Show key metrics
                                enhanced_df = enhanced_results["enhanced_dataset"]
                                if "composite_risk_score" in enhanced_df.columns:
                                    avg_risk = enhanced_df[
                                        "composite_risk_score"
                                    ].mean()
                                    max_risk = enhanced_df["composite_risk_score"].max()

                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric(
                                            "Average Risk Score", f"{avg_risk:.3f}"
                                        )
                                    with col2:
                                        st.metric(
                                            "Maximum Risk Score", f"{max_risk:.3f}"
                                        )
                                    with col3:
                                        risk_level = (
                                            "🟢 LOW"
                                            if avg_risk < 0.3
                                            else (
                                                "🟡 MEDIUM"
                                                if avg_risk < 0.7
                                                else "🔴 HIGH"
                                            )
                                        )
                                        st.metric("Risk Level", risk_level)
                            else:
                                st.error(
                                    "❌ Enhanced analysis failed. Check the logs for details."
                                )
                        else:
                            st.warning(
                                "Please select at least one symbol for enhanced analysis."
                            )

                    except Exception as e:
                        st.error(f"❌ Enhanced analysis error: {str(e)}")

            st.info(
                "💡 **Enhanced Features Include**: Advanced sentiment analysis, market regime detection, volatility scoring, and composite risk indicators!"
            )

        # Risk distribution and insights
        if "Risk_Label" in df.columns:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🥧 Risk Distribution")
                risk_counts = df["Risk_Label"].value_counts()
                total_days = len(df)

                fig = px.pie(
                    values=risk_counts.values,
                    names=[
                        "Safe Days 🟢" if x == 0 else "Risky Days 🔴"
                        for x in risk_counts.index
                    ],
                    title="How Often Was the Market Risky?",
                    color_discrete_map={
                        "Safe Days 🟢": "#28a745",
                        "Risky Days 🔴": "#dc3545",
                    },
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("📊 Risk Statistics")
                safe_days = risk_counts.get(0, 0)
                risky_days = risk_counts.get(1, 0)
                safe_pct = (safe_days / total_days) * 100
                risky_pct = (risky_days / total_days) * 100

                st.metric("Total Days Analyzed", f"{total_days:,}")
                st.metric("Safe Days 🟢", f"{safe_days:,} ({safe_pct:.1f}%)")
                st.metric("Risky Days 🔴", f"{risky_days:,} ({risky_pct:.1f}%)")

                # Risk insights
                if risky_pct > 30:
                    st.error(
                        "⚠️ **High Risk Period**: More than 30% of days were risky!"
                    )
                elif risky_pct > 20:
                    st.warning("🔶 **Moderate Risk**: About 1 in 5 days were risky")
                else:
                    st.success(
                        "✅ **Low Risk Period**: Most days were safe for investing"
                    )

        # Recent risk analysis
        if "Risk_Label" in df.columns and "Date" in df.columns:
            st.subheader("🕐 Recent Risk Trends")

            # Convert dates and get recent data
            df_recent = _normalize_datetime_column(df.copy(), "Date")
            df_recent = df_recent.sort_values("Date").tail(30)  # Last 30 days

            recent_risk = df_recent["Risk_Label"].mean()
            recent_risky_days = (df_recent["Risk_Label"] == 1).sum()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Last 30 Days Risk",
                    f"{recent_risk:.2f}",
                    delta=f"{recent_risk - df['Risk_Label'].mean():.2f} vs overall",
                )
            with col2:
                st.metric("Recent Risky Days", f"{recent_risky_days}/30")
            with col3:
                if recent_risk > 0.5:
                    st.error("🔴 Currently High Risk")
                elif recent_risk > 0.3:
                    st.warning("🟡 Currently Medium Risk")
                else:
                    st.success("🟢 Currently Low Risk")

    with tab_sentiment:
        st.subheader("Real-time Sentiment Analysis")

        # API Key input
        st.info("📰 Enter your GNews API key to fetch real-time news sentiment")
        gnews_api_key = st.text_input(
            "GNews API Key",
            type="password",
            help="Get your free API key from https://gnews.io/",
        )

        # Symbol selection
        available_symbols = (
            df["Symbol"].unique().tolist()
            if "Symbol" in df.columns
            else ["AAPL", "MSFT", "GOOGL"]
        )
        selected_symbols = st.multiselect(
            "Select symbols for news analysis",
            available_symbols,
            default=available_symbols[:3],
        )

        # Time range
        days_back = st.slider("Days of news to analyze", 1, 30, 7)

        if st.button("🔍 Fetch & Analyze News Sentiment", type="primary"):
            if not gnews_api_key:
                st.error("Please enter your GNews API key")
            elif not selected_symbols:
                st.error("Please select at least one symbol")
            else:
                with st.spinner("Fetching news and analyzing sentiment..."):
                    # Fetch news
                    news_df = fetch_gnews_sentiment(
                        selected_symbols, gnews_api_key, days_back
                    )

                    if news_df is not None and not news_df.empty:
                        st.success(f"✅ Fetched {len(news_df)} news articles")

                        # Analyze sentiment
                        news_with_sentiment = analyze_news_sentiment(news_df)

                        if news_with_sentiment is not None:
                            # Display sentiment summary with explanations
                            st.success(
                                f"✅ Analyzed {len(news_with_sentiment)} news articles"
                            )

                            col1, col2, col3, col4 = st.columns(4)

                            avg_sentiment = news_with_sentiment[
                                "sentiment_compound"
                            ].mean()
                            positive_pct = (
                                news_with_sentiment["sentiment_compound"] > 0.1
                            ).mean() * 100
                            negative_pct = (
                                news_with_sentiment["sentiment_compound"] < -0.1
                            ).mean() * 100
                            neutral_pct = 100 - positive_pct - negative_pct

                            with col1:
                                st.metric("Overall Mood", f"{avg_sentiment:.3f}")
                                if avg_sentiment > 0.1:
                                    st.success("😊 Positive")
                                elif avg_sentiment < -0.1:
                                    st.error("😟 Negative")
                                else:
                                    st.info("😐 Neutral")

                            with col2:
                                st.metric("Good News 📈", f"{positive_pct:.1f}%")
                            with col3:
                                st.metric("Bad News 📉", f"{negative_pct:.1f}%")
                            with col4:
                                st.metric("Neutral News ➡️", f"{neutral_pct:.1f}%")

                            # Sentiment interpretation
                            st.subheader("🧠 What Does This Mean?")

                            if avg_sentiment > 0.2:
                                st.success(
                                    "🚀 **Very Positive Sentiment**: News is overwhelmingly good! This often leads to price increases."
                                )
                            elif avg_sentiment > 0.05:
                                st.info(
                                    "📈 **Positive Sentiment**: More good news than bad. Generally bullish for prices."
                                )
                            elif avg_sentiment < -0.2:
                                st.error(
                                    "📉 **Very Negative Sentiment**: Lots of bad news. Prices may decline."
                                )
                            elif avg_sentiment < -0.05:
                                st.warning(
                                    "📊 **Negative Sentiment**: More bad news than good. Be cautious."
                                )
                            else:
                                st.info(
                                    "⚖️ **Neutral Sentiment**: News is balanced. Market may be stable."
                                )

                            # Sentiment scoring explanation
                            with st.expander("❓ How Are Sentiment Scores Calculated?"):
                                st.markdown(
                                    """
                                **Sentiment Score Range**: -1.0 (Very Negative) to +1.0 (Very Positive)
                                
                                **🔴 Negative (-1.0 to -0.05)**:
                                - Words like: "crash", "decline", "loss", "warning", "risk"
                                - Example: "Apple stock crashes due to poor earnings"
                                
                                **🟡 Neutral (-0.05 to +0.05)**:
                                - Factual reporting without emotional language
                                - Example: "Apple reports quarterly earnings results"
                                
                                **🟢 Positive (+0.05 to +1.0)**:
                                - Words like: "growth", "profit", "success", "strong", "excellent"
                                - Example: "Apple shows strong growth and exceeds expectations"
                                
                                **💡 Important**: Sentiment analysis looks at the language used, not just the topic!
                                """
                                )

                            # Combined sentiment and risk chart
                            sentiment_chart = create_sentiment_risk_chart(
                                news_with_sentiment, df
                            )
                            if sentiment_chart:
                                st.plotly_chart(
                                    sentiment_chart, use_container_width=True
                                )

                            # Sentiment by symbol with explanations
                            st.subheader("📊 Sentiment Breakdown by Company")

                            symbol_sentiment = (
                                news_with_sentiment.groupby("symbol")
                                .agg(
                                    {
                                        "sentiment_compound": ["mean", "std", "count"],
                                        "sentiment_positive": "mean",
                                        "sentiment_negative": "mean",
                                    }
                                )
                                .round(3)
                            )

                            symbol_sentiment.columns = [
                                "Avg_Sentiment",
                                "Volatility",
                                "Article_Count",
                                "Positive_Score",
                                "Negative_Score",
                            ]

                            # Add interpretation column
                            def interpret_sentiment(score):
                                if score > 0.2:
                                    return "Very Positive 🚀"
                                elif score > 0.05:
                                    return "Positive 📈"
                                elif score < -0.2:
                                    return "Very Negative 📉"
                                elif score < -0.05:
                                    return "Negative 📊"
                                else:
                                    return "Neutral ⚖️"

                            symbol_sentiment["Interpretation"] = symbol_sentiment[
                                "Avg_Sentiment"
                            ].apply(interpret_sentiment)

                            _render_dataframe(
                                symbol_sentiment, use_container_width=True
                            )

                            # Investment implications
                            st.subheader("💡 What This Means for Your Investments")
                            for symbol in selected_symbols:
                                if symbol in symbol_sentiment.index:
                                    score = symbol_sentiment.loc[
                                        symbol, "Avg_Sentiment"
                                    ]
                                    count = symbol_sentiment.loc[
                                        symbol, "Article_Count"
                                    ]

                                    if score > 0.1:
                                        st.success(
                                            f"📈 **{symbol}**: Positive news trend ({count} articles). Good time to consider buying."
                                        )
                                    elif score < -0.1:
                                        st.error(
                                            f"📉 **{symbol}**: Negative news trend ({count} articles). Consider waiting or selling."
                                        )
                                    else:
                                        st.info(
                                            f"⚖️ **{symbol}**: Neutral news ({count} articles). No strong signal either way."
                                        )

                            # Recent news headlines with better formatting
                            st.subheader("📰 Recent Headlines & Their Sentiment")

                            for symbol in selected_symbols:
                                symbol_news = news_with_sentiment[
                                    news_with_sentiment["symbol"] == symbol
                                ].head(3)
                                if not symbol_news.empty:
                                    st.markdown(f"### **{symbol}**")

                                    for i, (_, article) in enumerate(
                                        symbol_news.iterrows(), 1
                                    ):
                                        sentiment_score = article["sentiment_compound"]

                                        # Color and emoji based on sentiment
                                        if sentiment_score > 0.1:
                                            color = "success"
                                            emoji = "�"
                                            interpretation = "Bullish news"
                                        elif sentiment_score < -0.1:
                                            color = "error"
                                            emoji = "�"
                                            interpretation = "Bearish news"
                                        else:
                                            color = "info"
                                            emoji = "➡️"
                                            interpretation = "Neutral news"

                                        # Format with better styling
                                        with st.container():
                                            st.markdown(
                                                f"""
                                            **{i}. {emoji} {article['title']}**
                                            - **Sentiment**: {sentiment_score:.3f} ({interpretation})
                                            - **Source**: {article['source']} 
                                            - **Published**: {article['published_date']}
                                            """
                                            )

                                            # Add link if available
                                            if pd.notna(article["url"]):
                                                st.markdown(
                                                    f"[📖 Read full article]({article['url']})"
                                                )

                                            st.divider()
                    else:
                        st.warning(
                            "No news articles found for the selected symbols and time range"
                        )

        # Instructions
        st.subheader("How to get GNews API Key")
        st.markdown(
            """
        1. Visit [GNews.io](https://gnews.io/)
        2. Sign up for a free account
        3. Get your API key from the dashboard
        4. Free plan includes 100 requests per day
        """
        )

    with tab_data_explorer:
        st.subheader("Data Explorer")

        # Dataset overview
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Samples", len(df))
            st.metric("Features", len(df.columns))
        with col2:
            if "Symbol" in df.columns:
                st.metric("Symbols", df["Symbol"].nunique())
            if "Date" in df.columns:
                try:
                    date_col = _normalize_datetime_column(
                        df[["Date"]], "Date", dropna=False
                    )["Date"]
                    date_range = date_col.dt.date
                    st.metric("Date Range", f"{date_range.min()} to {date_range.max()}")
                except:
                    st.metric("Date Range", "Available")

        # Data preview
        st.subheader("Dataset Preview")
        _render_dataframe(df.head(100), use_container_width=True)

        # Data statistics
        st.subheader("Statistical Summary")
        _render_dataframe(df.describe(), use_container_width=True)

    with tab_model_details:
        st.subheader("Model Configuration Details")

        # Display model parameters
        for model_name, metrics in results.items():
            if model_name not in ["test_data"] and isinstance(metrics, dict):
                st.write(f"**{model_name.replace('_', ' ').title()}**")

                if "best_params" in metrics:
                    st.json(metrics["best_params"])

                if "classification_report" in metrics:
                    st.write("Classification Report:")
                    report_df = pd.DataFrame(
                        metrics["classification_report"]
                    ).transpose()
                    _render_dataframe(report_df, use_container_width=True)

                st.divider()


if __name__ == "__main__":
    main()
