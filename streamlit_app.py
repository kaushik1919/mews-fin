"""
Streamlit Frontend for Market Risk Early Warning System
Interactive dashboard with 4 ML models comparison
"""

import json
import os
import pickle
import shutil
import tempfile
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    from src.research.reporting import ResearchReportBuilder
except ImportError:  # pragma: no cover - research optional in UI
    ResearchReportBuilder = None  # type: ignore

warnings.filterwarnings("ignore")

# Import our system components
import sys

sys.path.append(".")
from datetime import datetime, timedelta

import requests

try:
    import mlflow  # type: ignore

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None  # type: ignore
    MLFLOW_AVAILABLE = False


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


def _render_dataframe(df: pd.DataFrame) -> None:
    formatted = _format_dataframe_for_display(df)
    st.dataframe(formatted)


def _render_plotly(fig: go.Figure, **kwargs: Any) -> None:
    """Render Plotly figures with responsive sizing and streamlined config."""

    default_config = {"displaylogo": False, "responsive": True}
    user_config = kwargs.pop("config", None)
    if isinstance(user_config, dict):
        default_config.update(user_config)
    elif user_config is not None:
        warnings.warn("Ignoring non-dict Plotly config passed to _render_plotly", RuntimeWarning)

    kwargs.pop("use_container_width", None)
    st.plotly_chart(fig, config=default_config, **kwargs)


def _render_download_buttons(df: pd.DataFrame, base_name: str, key_prefix: str) -> None:
    """Render paired CSV/HTML download buttons for a dataframe."""

    csv_data = df.to_csv(index=False).encode("utf-8")
    html_data = df.to_html(index=False)

    download_cols = st.columns(2)
    with download_cols[0]:
        st.download_button(
            "Download CSV",
            csv_data,
            file_name=f"{base_name}.csv",
            mime="text/csv",
            key=f"{key_prefix}_csv",
        )
    with download_cols[1]:
        st.download_button(
            "Download HTML",
            html_data,
            file_name=f"{base_name}.html",
            mime="text/html",
            key=f"{key_prefix}_html",
        )


def _sidebar_recommendation(text: str) -> None:
    """Display a consistent recommendation hint under sidebar controls."""

    st.sidebar.caption(f"**Recommended:** {text}")


def _format_metric_value(value: Optional[float], digits: int = 3) -> str:
    """Format numeric values for display, handling None and NaN gracefully."""

    if value is None:
        return "N/A"

    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return "N/A"
    except TypeError:
        pass

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _init_mlflow_client() -> Tuple[Optional["MlflowClient"], Optional[str], Optional[str]]:
    """Create an MLflow client pointed at the project's tracking directory."""

    try:
        import mlflow  # type: ignore
        from mlflow.tracking import MlflowClient  # type: ignore
    except ImportError:
        return None, None, "MLflow isn't installed. Run `pip install mlflow` to enable the experiment tracker."

    tracking_dir = Path(Config.MLFLOW_TRACKING_DIR)
    if not tracking_dir.exists() or not any(tracking_dir.glob("*")):
        return (
            None,
            None,
            f"No MLflow runs found yet. After executing the training pipeline, runs will appear under {tracking_dir}.",
        )

    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    client = MlflowClient()
    experiment = client.get_experiment_by_name(Config.MLFLOW_EXPERIMENT)
    if experiment is None:
        return (
            client,
            None,
            f"MLflow experiment '{Config.MLFLOW_EXPERIMENT}' not found. Run the pipeline to log experiments.",
        )

    return client, experiment.experiment_id, None


def _load_mlflow_runs_table(
    client: "MlflowClient", experiment_id: str, max_runs: int = 50
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]], List[str], Optional[str]]:
    """Fetch recent MLflow runs and convert them into a dataframe for display."""

    try:
        runs = client.search_runs(
            [experiment_id],
            filter_string="",
            order_by=["attributes.start_time DESC"],
            max_results=max_runs,
        )
    except Exception as exc:  # pragma: no cover - defensive against MLflow failures
        return pd.DataFrame(), {}, [], f"Failed to query MLflow runs: {exc}"

    rows: List[Dict[str, Any]] = []
    run_lookup: Dict[str, Dict[str, Any]] = {}
    ordered_ids: List[str] = []

    for run in runs:
        start_ts = (
            pd.Timestamp(run.info.start_time, unit="ms", tz="UTC").tz_convert(None)
            if run.info.start_time
            else None
        )
        end_ts = (
            pd.Timestamp(run.info.end_time, unit="ms", tz="UTC").tz_convert(None)
            if run.info.end_time
            else None
        )
        duration_seconds = (
            (end_ts - start_ts).total_seconds() if start_ts is not None and end_ts is not None else None
        )
        duration_display = (
            f"{duration_seconds / 60:.1f}" if duration_seconds is not None else "—"
        )

        metrics = dict(run.data.metrics)
        params = dict(run.data.params)
        tags = dict(run.data.tags)

        run_name = run.info.run_name or run.info.run_id[:8]
        primary_auc = metrics.get("ensemble.auc_score") or metrics.get("ensemble.auc")
        brier = metrics.get("ensemble.brier") or metrics.get("ensemble.brier_score")

        rows.append(
            {
                "Run ID": run.info.run_id,
                "Run": run_name,
                "Status": run.info.status.title(),
                "Start": start_ts.strftime("%Y-%m-%d %H:%M") if start_ts else "—",
                "Duration (min)": duration_display,
                "Ensemble AUC": _format_metric_value(primary_auc),
                "Brier": _format_metric_value(brier),
            }
        )

        run_lookup[run.info.run_id] = {
            "name": run_name,
            "run_id": run.info.run_id,
            "metrics": metrics,
            "params": params,
            "tags": tags,
            "status": run.info.status,
            "start": start_ts,
            "end": end_ts,
            "duration_seconds": duration_seconds,
        }

        ordered_ids.append(run.info.run_id)

    runs_df = pd.DataFrame(rows)
    return runs_df, run_lookup, ordered_ids, None


@st.cache_data(show_spinner=False)
def _load_mlflow_json_artifact(run_id: str, artifact_path: str) -> Optional[Dict[str, Any]]:
    """Download and parse a JSON artifact stored for a specific MLflow run."""

    try:
        import mlflow  # type: ignore
        from mlflow.tracking import MlflowClient  # type: ignore
    except ImportError:
        return None

    tracking_dir = Path(Config.MLFLOW_TRACKING_DIR)
    if not tracking_dir.exists():
        return None

    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    client = MlflowClient()

    temp_dir = Path(tempfile.mkdtemp(prefix="mlflow-artifact-"))
    try:
        local_path = Path(client.download_artifacts(run_id, artifact_path, str(temp_dir)))
        if local_path.is_dir():
            json_candidates = list(local_path.glob("*.json"))
            if json_candidates:
                local_path = json_candidates[0]
            else:
                return None

        if not local_path.exists():
            return None

        with open(local_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:  # pragma: no cover - artifact access issues
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@st.cache_resource(show_spinner=False)
def _load_mlflow_model(run_id: str, model_filename: str) -> Optional[Any]:
    """Load a pickled model artifact from MLflow."""

    try:
        import mlflow  # type: ignore
        from mlflow.tracking import MlflowClient  # type: ignore
    except ImportError:
        return None

    tracking_dir = Path(Config.MLFLOW_TRACKING_DIR)
    if not tracking_dir.exists():
        return None

    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    client = MlflowClient()

    temp_dir = Path(tempfile.mkdtemp(prefix="mlflow-model-"))
    try:
        local_path = Path(
            client.download_artifacts(
                run_id, f"artifacts/models/{model_filename}", str(temp_dir)
            )
        )
        if not local_path.exists():
            return None

        with open(local_path, "rb") as handle:
            return pickle.load(handle)
    except Exception:  # pragma: no cover - artifact access issues
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _prepare_model_sample(
    df: pd.DataFrame, model: Any, target_column: str = "Risk_Label", sample_size: int = 200
) -> Optional[pd.DataFrame]:
    """Align the dataframe to the model's feature ordering and subsample for SHAP."""

    feature_order = getattr(model, "feature_names_in_", None)
    feature_matrix = prepare_feature_matrix(df, target_column=target_column, feature_order=feature_order)
    if feature_matrix.empty:
        return None

    sample_n = min(sample_size, len(feature_matrix))
    if sample_n == 0:
        return None

    return feature_matrix.sample(n=sample_n, random_state=42)


def _compute_run_shap_global(
    run_id: str, model_name: str, df: pd.DataFrame, sample_size: int = 200
) -> Optional[pd.DataFrame]:
    """Compute global SHAP values for a model artifact stored in MLflow."""

    model = _load_mlflow_model(run_id, f"{model_name}_model.pkl")
    if model is None:
        return None

    sample = _prepare_model_sample(df, model, sample_size=sample_size)
    if sample is None:
        return None

    shap_df = compute_global_shap_importance(model, sample)
    if shap_df is None or shap_df.empty:
        return None

    return shap_df


from src.backtester import RiskBacktester
from src.metrics import compute_cews_score
from src.xai_utils import (
    LIME_AVAILABLE,
    SHAP_AVAILABLE,
    compute_global_shap_importance,
    compute_lime_explanation,
    compute_local_shap_explanation,
)

from src.case_studies import CaseStudyRunner, CaseStudyScenario, PREDEFINED_CASE_STUDIES
from src.config import Config
from src.experiments import ExperimentConfig, ExperimentManager
from src.multimodal_fusion import FusionInputs, MultiModalFeatureFusion
from src.robustness import PerturbationConfig, RobustnessEvaluator

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from mlflow.tracking import MlflowClient  # type: ignore

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
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] button {
        width: 100%;
    }
    div[data-testid="stPlotlyChart"] iframe {
        width: 100% !important;
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


def reset_streamlit_caches() -> None:
    """Clear cached readers so downstream tabs reload after pipeline runs."""

    cache_functions = [
        load_predictions_cache,
        run_cached_backtest,
        prepare_feature_matrix,
        _load_mlflow_json_artifact,
    ]

    for cache_fn in cache_functions:
        clear = getattr(cache_fn, "clear", None)
        if callable(clear):
            clear()


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


PIPELINE_STAGE_OPTIONS = [
    "Data",
    "Preprocessing",
    "Sentiment",
    "Fusion",
    "Training",
    "Evaluation",
    "Robustness",
]

MODEL_NAME_MAP = {
    "Random Forest": "random_forest",
    "Logistic Regression": "logistic_regression",
    "XGBoost": "xgboost",
    "SVM": "svm",
}


def load_sentiment_cache() -> Optional[pd.DataFrame]:
    sentiment_path = Path("data/sentiment_aggregated.csv")
    if not sentiment_path.exists():
        return None

    try:
        sentiment_df = pd.read_csv(sentiment_path)
        for candidate in ("Date", "published_date"):
            if candidate in sentiment_df.columns:
                sentiment_df = _normalize_datetime_column(sentiment_df, candidate)
                if candidate != "Date":
                    sentiment_df = sentiment_df.rename(columns={candidate: "Date"})
                break
        return sentiment_df
    except Exception:
        return None


def load_finbert_sentiment_cache() -> Optional[pd.DataFrame]:
    finbert_path = Path("data/sec_sentiment.csv")
    if not finbert_path.exists():
        return None

    try:
        finbert_df = pd.read_csv(finbert_path)
        rename_map = {}
        if "date" in finbert_df.columns:
            rename_map["date"] = "Date"
        if "symbol" in finbert_df.columns:
            rename_map["symbol"] = "Symbol"
        finbert_df = finbert_df.rename(columns=rename_map)
        if "Date" in finbert_df.columns:
            finbert_df = _normalize_datetime_column(finbert_df, "Date")

        sentiment_cols = [
            col
            for col in finbert_df.columns
            if any(
                keyword in col
                for keyword in (
                    "sentiment",
                    "sentiment_label",
                    "sentiment_intensity",
                    "combined_sec",
                )
            )
        ]
        base_cols = [col for col in ("Symbol", "Date") if col in finbert_df.columns]
        projection = base_cols + sentiment_cols
        finbert_df = finbert_df[projection].copy()
        return finbert_df
    except Exception:
        return None


def prepare_sentiment_dataframe(
    sentiment_choice: str,
    selected_symbols: Iterable[str],
) -> Optional[pd.DataFrame]:
    sentiment_choice_normalized = sentiment_choice.lower()
    if sentiment_choice_normalized == "finbert":
        sentiment_df = load_finbert_sentiment_cache()
        if sentiment_df is not None and not sentiment_df.empty:
            numeric_cols = [
                col
                for col in sentiment_df.columns
                if col not in {"Symbol", "Date"}
                and pd.api.types.is_numeric_dtype(sentiment_df[col])
            ]
            if numeric_cols:
                sentiment_df = (
                    sentiment_df.groupby([col for col in ["Symbol", "Date"] if col in sentiment_df.columns])[numeric_cols]
                    .mean()
                    .reset_index()
                )
    else:
        sentiment_df = load_sentiment_cache()

    if sentiment_df is None or sentiment_df.empty:
        return sentiment_df

    if selected_symbols:
        if "Symbol" in sentiment_df.columns:
            sentiment_df = sentiment_df[sentiment_df["Symbol"].isin(selected_symbols)]

    return sentiment_df


def load_news_cache(max_rows: int = 500) -> Optional[pd.DataFrame]:
    news_path = Path("data/news_data.csv")
    if not news_path.exists():
        return None

    try:
        news_df = pd.read_csv(news_path)
        if not news_df.empty:
            if "published_date" in news_df.columns:
                news_df = _normalize_datetime_column(news_df, "published_date")
                news_df = news_df.rename(columns={"published_date": "Date"})
            elif "Date" in news_df.columns:
                news_df = _normalize_datetime_column(news_df, "Date")

            if "symbol" in news_df.columns and "Symbol" not in news_df.columns:
                news_df = news_df.rename(columns={"symbol": "Symbol"})

            subset_cols = [col for col in ["Symbol", "Date", "title", "description", "content"] if col in news_df.columns]
            if subset_cols:
                news_df = news_df[subset_cols]
            if len(news_df) > max_rows:
                news_df = news_df.sample(max_rows, random_state=42)
        return news_df
    except Exception:
        return None


def load_latest_robustness_report() -> Optional[Dict[str, Any]]:
    root = Path("outputs/robustness")
    if not root.exists():
        return None

    candidates = sorted(root.glob("robustness_*/robustness_report.json"))
    if not candidates:
        fallback = root / "robustness_report.json"
        if fallback.exists():
            candidates = [fallback]

    if not candidates:
        return None

    try:
        with open(candidates[-1], "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def run_light_robustness(
    df: pd.DataFrame,
    ensemble_choice: str,
    noise_level: float = 0.05,
    delay_days: int = 1,
) -> Optional[Dict[str, Any]]:
    if df.empty:
        return None

    sample = df.copy()
    if len(sample) > 500:
        sample = sample.sample(500, random_state=42)

    try:
        evaluator = RobustnessEvaluator(dataset=sample, output_root=Path("outputs/robustness/streamlit"))
        perturbations = [
            PerturbationConfig(
                name=f"noise_{int(noise_level * 100)}pct",
                kind="noise",
                params={"noise_level": float(noise_level)},
            ),
            PerturbationConfig(
                name=f"delay_{delay_days}d",
                kind="delay",
                params={"delay_days": int(delay_days)},
            ),
        ]

        report = evaluator.run(perturbations=perturbations, auditor=None)
        return report.to_dict()
    except Exception as exc:
        st.warning(f"Robustness evaluation unavailable: {exc}")
        return None


def run_fusion_preview(
    df: pd.DataFrame,
    fusion_choice: str,
    news_df: Optional[pd.DataFrame],
) -> Optional[Dict[str, Any]]:
    if fusion_choice == "none":
        return None

    required_cols = [col for col in ["Symbol", "Date"] if col in df.columns]
    if len(required_cols) < 2:
        return None

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    base_cols = required_cols + numeric_cols[:15]
    tabular = df[base_cols].dropna(subset=[required_cols[0], required_cols[1]])
    if tabular.empty:
        return None

    try:
        fusion_engine = MultiModalFeatureFusion(
            fusion_strategy=fusion_choice if fusion_choice != "auto" else "concat",
            enable_gnn=False,
        )
        fusion_inputs = FusionInputs(tabular_features=tabular, news_df=news_df)
        fused = fusion_engine.fuse(fusion_inputs)
        new_columns = [col for col in fused.columns if col not in tabular.columns]
        return {
            "fused": fused.head(200),
            "new_columns": new_columns[:15],
            "strategy": fusion_engine.fusion_strategy,
        }
    except Exception as exc:
        st.warning(f"Fusion preview failed: {exc}")
        return None


def execute_stage_controls(
    selected_stages: Iterable[str],
    selected_models: Iterable[str],
    fusion_choice: str,
    ensemble_choice: str,
    df: pd.DataFrame,
    results: Dict[str, Any],
    sentiment_df: Optional[pd.DataFrame],
    risk_threshold: float,
    cews_threshold: float,
    robustness_config: Dict[str, Any],
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> list[Dict[str, Any]]:
    stage_outputs: list[Dict[str, Any]] = []
    stage_order = (
        "Data",
        "Preprocessing",
        "Sentiment",
        "Fusion",
        "Training",
        "Evaluation",
        "Robustness",
    )
    tracked_stages = [stage for stage in stage_order if stage in selected_stages]
    total_tracked = len(tracked_stages)
    completed_stages = 0

    def report_progress(stage_name: str) -> None:
        nonlocal completed_stages
        if stage_name not in tracked_stages or total_tracked == 0:
            return
        completed_stages += 1
        if progress_callback:
            progress_callback(stage_name, completed_stages, total_tracked)

    if "Data" in selected_stages:
        sample = df.head(500)
        summary_metrics = []
        summary_metrics.append({"label": "Rows", "value": f"{len(df):,}"})
        if "Symbol" in df.columns:
            summary_metrics.append({"label": "Symbols", "value": f"{df['Symbol'].nunique():,}"})
        if "Date" in df.columns:
            normalized = _normalize_datetime_column(df[["Date"]].copy())
            start_date = normalized["Date"].min()
            end_date = normalized["Date"].max()
            if pd.notna(start_date) and pd.notna(end_date):
                summary_metrics.append({"label": "Date Range", "value": f"{start_date.date()} → {end_date.date()}"})

        if "Risk_Probability" in df.columns:
            summary_metrics.append(
                {
                    "label": "Avg Risk Probability",
                    "value": f"{df['Risk_Probability'].mean():.3f}",
                    "delta": f"Threshold {risk_threshold:.2f}",
                }
            )

        if {"Risk_Label", "Risk_Probability"}.issubset(df.columns):
            try:
                cews_result = compute_cews_score(
                    df,
                    probability_col="Risk_Probability",
                    label_col="Risk_Label",
                    threshold=cews_threshold,
                )
                alert_series = df["Risk_Probability"] >= risk_threshold
                alert_rate = float(alert_series.mean()) if len(alert_series) else 0.0
                if not np.isfinite(alert_rate):
                    alert_rate = 0.0
                summary_metrics.append(
                    {
                        "label": "CEWS Score",
                        "value": f"{cews_result.score:.2f}",
                        "delta": f"Alerts {alert_rate:.0%}",
                    }
                )
            except Exception:
                pass

        stage_outputs.append(
            {
                "title": "Data Stage – Overview",
                "messages": ["Previewing cached integrated dataset. Adjust the other stages to recompute downstream analytics."],
                "metrics": summary_metrics,
                "dataframe": sample[[col for col in sample.columns if col in ["Date", "Symbol"] + sample.select_dtypes(include=[np.number]).columns.tolist()[:8]]],
            }
        )
        report_progress("Data")

    if "Preprocessing" in selected_stages:
        missing = (
            df.isna()
            .mean()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
            .rename(columns={"index": "Feature", 0: "Missing_Rate"})
        )
        if "Missing_Rate" in missing.columns:
            missing["Missing_Rate"] = (missing["Missing_Rate"] * 100).round(2)

        stats = (
            df.select_dtypes(include=[np.number])
            .describe()
            .transpose()[["mean", "std"]]
            .reset_index()
            .rename(columns={"index": "Feature", "mean": "Mean", "std": "Std Dev"})
            .head(15)
        )

        stage_outputs.append(
            {
                "title": "Preprocessing Stage – Data Quality Checks",
                "messages": [
                    "Missing-value scan and quick numeric stats help you confirm inputs before modeling."
                ],
                "tables": [
                    {"caption": "Highest Missing Rates (%)", "data": missing},
                    {"caption": "Sample Numeric Feature Stats", "data": stats},
                ],
            }
        )
        report_progress("Preprocessing")

    if "Sentiment" in selected_stages:
        if sentiment_df is not None and not sentiment_df.empty:
            numeric_cols = [col for col in sentiment_df.columns if col not in {"Date", "Symbol"} and pd.api.types.is_numeric_dtype(sentiment_df[col])]
            plot_col = numeric_cols[0] if numeric_cols else None
            sentiment_fig = None
            sentiment_messages = [
                "Sentiment aggregates combine news and filings. Positive swings above zero often precede bullish periods; sustained negatives warn of stress."
            ]

            if plot_col:
                sentiment_preview = sentiment_df.copy()

                date_candidate = next(
                    (candidate for candidate in ("Date", "date", "published_date") if candidate in sentiment_preview.columns),
                    None,
                )

                if date_candidate and date_candidate != "Date":
                    sentiment_preview = sentiment_preview.rename(columns={date_candidate: "Date"})

                if "Date" in sentiment_preview.columns:
                    sentiment_preview = _normalize_datetime_column(sentiment_preview, "Date")
                    sentiment_preview = sentiment_preview.dropna(subset=["Date"])
                    if not sentiment_preview.empty:
                        try:
                            timeline_preview = (
                                sentiment_preview.groupby("Date")[plot_col].mean().reset_index()
                            )
                            if not timeline_preview.empty:
                                sentiment_fig = px.line(
                                    timeline_preview,
                                    x="Date",
                                    y=plot_col,
                                    title=f"Average {plot_col} Over Time",
                                )
                        except KeyError:
                            sentiment_messages.append(
                                "Daily trend chart unavailable because the sentiment dataset does not include a clean Date column after normalization."
                            )
                else:
                    sentiment_messages.append(
                        "Daily trend chart unavailable because the loaded sentiment snapshot does not include a Date column."
                    )

            stage_outputs.append(
                {
                    "title": "Sentiment Stage – Aggregated Signals",
                    "messages": sentiment_messages,
                    "figure": sentiment_fig,
                    "dataframe": sentiment_df.head(200),
                }
            )
        else:
            stage_outputs.append(
                {
                    "title": "Sentiment Stage – Aggregated Signals",
                    "messages": ["No sentiment data available for the current configuration. Re-run the pipeline via CLI to populate the sentiment cache."],
                }
            )
        report_progress("Sentiment")

    if "Fusion" in selected_stages:
        news_df = load_news_cache()
        fusion_payload = run_fusion_preview(df, fusion_choice, news_df)
        if fusion_payload:
            new_cols = fusion_payload["new_columns"]
            caption = (
                "New fused features (first 15 columns shown)" if new_cols else "Fusion produced no additional columns; showing blended sample"
            )
            stage_outputs.append(
                {
                    "title": f"Fusion Stage – Strategy: {fusion_payload['strategy'].replace('_', ' ').title()}",
                    "messages": [
                        "Fused outputs blend tabular indicators with optional news embeddings. Preview below reflects the selected strategy."
                    ],
                    "tables": [
                        {"caption": caption, "data": pd.DataFrame({"Feature": new_cols}) if new_cols else None}
                    ],
                    "dataframe": fusion_payload["fused"],
                }
            )
        else:
            stage_outputs.append(
                {
                    "title": "Fusion Stage",
                    "messages": ["Fusion preview unavailable. Ensure Symbol and Date columns exist and at least one fusion strategy is selected."],
                }
            )
        report_progress("Fusion")

    if "Training" in selected_stages:
        training_metrics = []
        detail_rows = []
        for label in selected_models:
            key = MODEL_NAME_MAP.get(label)
            if key and key in results:
                metrics = results[key]
                training_metrics.append(
                    {
                        "label": f"{label} AUC",
                        "value": f"{metrics.get('auc_score', 0):.3f}",
                        "delta": f"Acc {metrics.get('test_accuracy', 0):.3f}",
                    }
                )
                detail_rows.append(
                    {
                        "Model": label,
                        "Accuracy": f"{metrics.get('test_accuracy', 0):.4f}",
                        "AUC": f"{metrics.get('auc_score', 0):.4f}",
                        "Threshold": f"{metrics.get('optimal_threshold', 0.5):.2f}",
                    }
                )

        if "ensemble" in results:
            ensemble_metrics = results["ensemble"]
            training_metrics.append(
                {
                    "label": "Ensemble AUC",
                    "value": f"{ensemble_metrics.get('auc_score', 0):.3f}",
                    "delta": f"Acc {ensemble_metrics.get('test_accuracy', 0):.3f}",
                }
            )

        stage_outputs.append(
            {
                "title": "Training Stage – Model Performance",
                "metrics": training_metrics if training_metrics else None,
                "dataframe": pd.DataFrame(detail_rows) if detail_rows else None,
                "messages": [
                    "Metrics pulled from the latest cached training run. Adjust selections to focus on specific model families."
                ],
            }
        )
        report_progress("Training")

    if "Evaluation" in selected_stages:
        y_true_pred = get_test_predictions(results)
        if y_true_pred:
            from sklearn.metrics import precision_score, recall_score, roc_auc_score

            y_true, _, y_prob = y_true_pred
            y_pred = (y_prob >= risk_threshold).astype(int)
            evaluation_metrics = [
                {"label": "AUC", "value": f"{roc_auc_score(y_true, y_prob):.3f}"},
                {"label": "Precision", "value": f"{precision_score(y_true, y_pred, zero_division=0):.3f}"},
                {"label": "Recall", "value": f"{recall_score(y_true, y_pred, zero_division=0):.3f}"},
            ]
            confusion_fig = create_confusion_matrix_plot(y_true, y_pred)
            stage_outputs.append(
                {
                    "title": "Evaluation Stage – Test Set Diagnostics",
                    "metrics": evaluation_metrics,
                    "figure": confusion_fig,
                    "messages": ["Confusion matrix and summary metrics calculated from cached test predictions."],
                }
            )
        else:
            stage_outputs.append(
                {
                    "title": "Evaluation Stage",
                    "messages": ["Test predictions were not cached. Re-run training to enable evaluation diagnostics."],
                }
            )
        report_progress("Evaluation")

    if "Robustness" in selected_stages:
        noise_level = float(robustness_config.get("noise", 0.05))
        delay_days = int(robustness_config.get("delay", 1))
        robustness_report = load_latest_robustness_report() or run_light_robustness(
            df,
            ensemble_choice,
            noise_level=noise_level,
            delay_days=delay_days,
        )
        if robustness_report:
            baseline_metrics = robustness_report.get("baseline_metrics", {}).get("ensemble", {})
            perturbations = robustness_report.get("perturbations", [])
            robustness_metrics = []
            if baseline_metrics:
                for metric_key in ["auc_score", "cews_score", "test_accuracy"]:
                    if metric_key in baseline_metrics:
                        robustness_metrics.append(
                            {
                                "label": metric_key.replace("_", " ").title(),
                                "value": f"{baseline_metrics[metric_key]:.3f}",
                                "delta": (
                                    f"Noise {noise_level:.0%}" if metric_key == "auc_score" else None
                                ),
                            }
                        )

            perturbation_df = None
            if perturbations:
                records = []
                for entry in perturbations:
                    deltas = entry.get("deltas", {})
                    records.append(
                        {
                            "Scenario": entry.get("name"),
                            "Kind": entry.get("kind"),
                            "ΔAUC": f"{deltas.get('auc_score', 0.0):+.3f}",
                            "ΔAccuracy": f"{deltas.get('test_accuracy', 0.0):+.3f}",
                            "ΔCEWS": f"{deltas.get('cews_score', 0.0):+.3f}",
                            "Noise": f"{noise_level:.0%}",
                            "Delay": f"{delay_days}d",
                        }
                    )
                perturbation_df = pd.DataFrame(records)

            stage_outputs.append(
                {
                    "title": "Robustness Stage – Stress Test Summary",
                    "metrics": robustness_metrics if robustness_metrics else None,
                    "dataframe": perturbation_df,
                    "messages": [
                        "Examines how core metrics shift under noise and reporting delays. Positive deltas mean resilience; negative values highlight vulnerabilities."
                    ],
                }
            )
        else:
            stage_outputs.append(
                {
                    "title": "Robustness Stage",
                    "messages": ["No robustness reports available and on-the-fly evaluation failed."],
                }
            )
        report_progress("Robustness")

    return stage_outputs


def render_stage_outputs(stage_outputs: Iterable[Dict[str, Any]]) -> None:
    for entry in stage_outputs:
        st.markdown(f"### {entry['title']}")
        for message in entry.get("messages", []):
            st.markdown(message)

        metrics = entry.get("metrics")
        if metrics:
            cols = st.columns(len(metrics))
            for col, metric in zip(cols, metrics):
                delta_value = metric.get("delta") or ""
                col.metric(metric.get("label", ""), metric.get("value", ""), delta_value)

        for table in entry.get("tables", []) or []:
            data = table.get("data")
            if isinstance(data, pd.DataFrame) and not data.empty:
                st.markdown(f"**{table.get('caption', 'Table')}**")
                _render_dataframe(data)

        df_preview = entry.get("dataframe")
        if isinstance(df_preview, pd.DataFrame) and not df_preview.empty:
            _render_dataframe(df_preview.head(100))

        figure = entry.get("figure")
        if figure is not None:
            _render_plotly(figure)


def run_benchmarks(df: pd.DataFrame, ensemble_choice: str) -> Optional[Dict[str, Any]]:
    if df.empty:
        return None

    sample = df.copy()
    if len(sample) > 300:
        sample = sample.sample(300, random_state=42)

    config = ExperimentConfig(
        name="streamlit_benchmarks",
        baselines=[
            {"type": "value_at_risk", "params": {"window": 20}},
            {"type": "garch", "params": {"confidence": 0.95}},
        ],
        mews={
            "enabled": True,
            "regime_adaptive": {"enabled": ensemble_choice == "Regime Adaptive"},
            "test_size": 0.25,
        },
        output_dir="outputs/experiments",
    )

    try:
        manager = ExperimentManager(config=config, dataframe=sample)
        return manager.run()
    except Exception as exc:
        st.warning(f"Benchmark run failed: {exc}")
        return None


def display_benchmark_results(summary: Dict[str, Any]) -> None:
    st.markdown("### Benchmark Results")
    st.markdown(f"**Experiment:** {summary.get('experiment', 'N/A')}")
    runs = summary.get("runs", [])
    for run in runs:
        st.markdown(f"#### Segment: {run.get('label', 'baseline')}")
        baseline_entries = run.get("baselines", [])
        if baseline_entries:
            baseline_df = pd.DataFrame(baseline_entries)
            _render_dataframe(baseline_df)
        if run.get("combined_baselines"):
            st.caption(f"Combined predictions saved to: {run['combined_baselines']}")
        if "mews_model" in run:
            mews_info = run["mews_model"]
            metrics_path = mews_info.get("metrics")
            st.markdown("- MEWS predictions stored at: `{}`".format(mews_info.get("predictions", "-")))
            if metrics_path:
                st.markdown(f"- Metrics JSON: `{metrics_path}`")


def run_case_studies(selected_slugs: Iterable[str]) -> list[Any]:
    runner = CaseStudyRunner()
    scenarios = {scenario.slug: scenario for scenario in PREDEFINED_CASE_STUDIES}
    results: list[Any] = []
    for slug in selected_slugs:
        scenario = scenarios.get(slug)
        if scenario is None:
            continue
        try:
            results.append(runner.run_case_study(scenario))
        except Exception as exc:
            results.append({"scenario": scenario, "error": str(exc)})
    return results


def display_case_study_results(results: Iterable[Any]) -> None:
    for result in results:
        if isinstance(result, dict) and "error" in result:
            scenario_name = getattr(result.get("scenario"), "name", "Unknown Scenario")
            st.warning(f"{scenario_name}: {result['error']}")
            continue

        scenario = result.scenario
        st.markdown(f"#### {scenario.name}")
        metrics_columns = st.columns(4)
        metrics_columns[0].metric("Observations", f"{result.total_observations:,}")
        metrics_columns[1].metric("Warnings", str(result.warning_events))
        metrics_columns[2].metric("Downturns", str(result.downturn_events))
        metrics_columns[3].metric("Overlap", str(result.combined_events))

        if result.plot_path and result.plot_path.exists():
            st.image(str(result.plot_path), caption="Risk vs Downturn")

        if result.top_features:
            feature_df = pd.DataFrame(result.top_features, columns=["Feature", "Frequency"])
            _render_dataframe(feature_df)

        if result.report_path and result.report_path.exists():
            st.caption(f"Markdown report saved to: {result.report_path.as_posix()}")


@st.cache_data(show_spinner=False)
def load_predictions_cache() -> Optional[pd.DataFrame]:
    predictions_path = Path("outputs/risk_predictions.csv")
    if not predictions_path.exists():
        return None

    try:
        predictions = pd.read_csv(predictions_path)
        predictions = _normalize_datetime_column(predictions, "Date")
        if "Actual_Risk_Label" in predictions.columns and "Risk_Label" not in predictions.columns:
            predictions = predictions.rename(columns={"Actual_Risk_Label": "Risk_Label"})
        return predictions
    except Exception:
        return None


def prepare_case_study_dataset(
    base_df: pd.DataFrame,
    scenario: CaseStudyScenario,
    predictions_df: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    working = base_df.copy()
    working = _normalize_datetime_column(working, "Date")

    merged = working
    if predictions_df is not None and not predictions_df.empty:
        predictions = predictions_df.copy()
        predictions = _normalize_datetime_column(predictions, "Date")
        predictions = predictions.drop_duplicates(subset=["Date", "Symbol"], keep="last")
        merged = merged.merge(
            predictions[[col for col in predictions.columns if col in {"Date", "Symbol", "Risk_Prediction", "Risk_Probability", "Risk_Label"}]],
            on=["Date", "Symbol"],
            how="left",
            suffixes=("", "_pred"),
        )

        prob_columns = [col for col in merged.columns if col.startswith("Risk_Probability")]
        if prob_columns:
            merged["Risk_Probability"] = merged[prob_columns].bfill(axis=1).iloc[:, 0]
            for col in prob_columns:
                if col != "Risk_Probability":
                    merged = merged.drop(columns=col)

        if "Risk_Prediction_pred" in merged.columns and "Risk_Prediction" not in merged.columns:
            merged = merged.rename(columns={"Risk_Prediction_pred": "Risk_Prediction"})

    start = pd.Timestamp(scenario.start_date)
    end = pd.Timestamp(scenario.end_date)

    subset = merged[(merged["Date"] >= start) & (merged["Date"] <= end)].copy()
    if scenario.symbols:
        subset = subset[subset["Symbol"].isin(scenario.symbols)]

    if subset.empty:
        return subset, pd.DataFrame()

    subset = subset.sort_values(["Date", "Symbol"])  # type: ignore[arg-type]

    if "Close" in subset.columns:
        subset["Close"] = subset["Close"].astype(float)
        subset["Rolling_Max_Close"] = subset.groupby("Symbol")["Close"].cummax()
        subset["Drawdown"] = subset["Close"] / subset["Rolling_Max_Close"] - 1.0
    elif "Returns" in subset.columns:
        subset["Drawdown"] = subset.groupby("Symbol")["Returns"].cumsum()
    else:
        subset["Drawdown"] = 0.0

    if "Risk_Probability" not in subset.columns or subset["Risk_Probability"].isna().all():
        subset["Risk_Probability"] = subset.groupby("Symbol")["Risk_Label"].transform(
            lambda s: s.rolling(window=3, min_periods=1).mean()
        )

    subset["Risk_Probability"] = subset["Risk_Probability"].clip(0, 1)
    subset = subset.drop(columns=["Rolling_Max_Close"], errors="ignore")

    timeline = (
        subset.groupby("Date")
        .agg(
            Risk_Probability=("Risk_Probability", "mean"),
            Drawdown=("Drawdown", "mean"),
            Warning_Rate=("Risk_Label", "mean"),
        )
        .reset_index()
    )

    if not timeline.empty:
        timeline["Drawdown"] = timeline["Drawdown"].fillna(0.0)
        timeline["Risk_Probability"] = timeline["Risk_Probability"].fillna(0.0)
        timeline["Drawdown_pct"] = timeline["Drawdown"].clip(upper=0)

    return subset, timeline


def create_case_study_plot(
    timeline: pd.DataFrame,
    scenario: CaseStudyScenario,
    risk_threshold: float,
) -> Optional[go.Figure]:
    if timeline.empty:
        return None

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=timeline["Date"],
            y=timeline["Risk_Probability"],
            name="Avg Risk Probability",
            mode="lines",
            line=dict(color="#dc2626", width=3),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Risk: %{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=timeline["Date"],
            y=timeline["Drawdown"],
            name="Avg Drawdown",
            mode="lines",
            line=dict(color="#2563eb", width=3),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:.1%}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.add_hline(
        y=risk_threshold,
        line_width=2,
        line_dash="dot",
        line_color="#f97316",
        annotation_text=f"Risk Threshold {risk_threshold:.2f}",
        annotation_position="top left",
    )

    for marker_date, label in scenario.marker_dates():
        try:
            marker_ts = pd.Timestamp(marker_date)
        except Exception:
            continue
        fig.add_vline(
            x=marker_ts,
            line_dash="dash",
            line_color="#94a3b8",
            opacity=0.6,
        )
        fig.add_annotation(
            x=marker_ts,
            y=1.02,
            xref="x",
            yref="paper",
            text=label,
            showarrow=False,
            bgcolor="rgba(148,163,184,0.2)",
            bordercolor="#94a3b8",
            borderwidth=1,
            yanchor="bottom",
            font=dict(size=11),
        )

    fig.update_yaxes(
        title_text="Avg Risk Probability",
        range=[0, 1],
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Avg Drawdown",
        tickformat=".0%",
        secondary_y=True,
    )
    fig.update_layout(
        title=f"{scenario.name} – Risk vs Market Drawdown",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.2, x=0),
        margin=dict(t=70, b=80, l=70, r=70),
    )
    return fig

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

    st.sidebar.markdown("### Pipeline Controls")

    available_symbols = (
        sorted(df["Symbol"].dropna().unique()) if "Symbol" in df.columns else []
    )
    default_symbols = available_symbols[:5] if len(available_symbols) > 5 else available_symbols
    selected_symbols = st.sidebar.multiselect(
        "Select tickers",
        options=available_symbols,
        default=default_symbols,
        help=(
            "Limit the analysis universe to tickers already present in the cached dataset. "
            "Impacts every tab by filtering data, so selecting fewer symbols speeds up charts while "
            "narrowing coverage."
        ),
    )
    _sidebar_recommendation("Start with 3–5 highly traded tickers to balance speed and context.")

    sentiment_choice = st.sidebar.selectbox(
        "Sentiment model",
        options=["VADER", "FinBERT"],
        help=(
            "Determines which NLP model scores headlines and filings. VADER is fast but generic; FinBERT "
            "captures finance-specific tone and changes downstream fusion features accordingly."
        ),
    )
    _sidebar_recommendation("FinBERT provides the most reliable finance-domain sentiment.")

    selected_stages = st.sidebar.multiselect(
        "Pipeline stages",
        PIPELINE_STAGE_OPTIONS,
        default=PIPELINE_STAGE_OPTIONS,
        help=(
            "Toggle which processing stages run when you orchestrate the pipeline. Disabling stages skips their "
            "analytics but shortens execution when you only need specific outputs."
        ),
    )
    _sidebar_recommendation("Keep all stages enabled for a full-system health check.")

    model_labels = list(MODEL_NAME_MAP.keys())
    selected_models = st.sidebar.multiselect(
        "Models to monitor",
        model_labels,
        default=model_labels,
        help=(
            "Choose which trained model families appear in the Model Results tab and evaluation comparisons. "
            "Removing models hides their metrics and speeds up plotting for lightweight reviews."
        ),
    )
    _sidebar_recommendation("Track the full suite when benchmarking the ensemble’s uplift.")

    fusion_options = ["concat", "cross_attention", "gated"]
    fusion_display = {
        "concat": "Concat",
        "cross_attention": "Cross Attention",
        "gated": "Gated",
    }
    fusion_choice = st.sidebar.selectbox(
        "Fusion strategy",
        fusion_options,
        index=0,
        format_func=lambda key: fusion_display.get(key, key.title()),
        help=(
            "Controls how tabular indicators, sentiment features, and graph signals are merged before modeling. "
            "Concat is the fastest baseline; cross-attention and gated fusion better capture cross-modal interplay."
        ),
    )
    _sidebar_recommendation("Use Concat for quick iteration; switch to Cross Attention for richer experiments.")

    ensemble_options = ["Static", "Regime Adaptive"]
    ensemble_choice = st.sidebar.selectbox(
        "Ensemble strategy",
        ensemble_options,
        index=0,
        help=(
            "Select how base learners are blended. Static keeps fixed weights for consistency; Regime Adaptive "
            "shifts weights with volatility regimes to chase extra lift at the cost of complexity."
        ),
    )
    _sidebar_recommendation("Begin with Static weighting, then trial Regime Adaptive once metrics are stable.")

    st.sidebar.markdown("### Thresholds")
    risk_threshold = st.sidebar.slider(
        "Risk probability threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help=(
            "Defines the probability cutoff that converts predictions into high-risk alerts. Lower values fire more "
            "warnings (higher recall) while higher values cut false positives but may miss early signals."
        ),
    )
    _sidebar_recommendation("0.50 is a balanced starting point; lower it during stress testing for sensitivity.")

    cews_threshold = st.sidebar.slider(
        "CEWS alert threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.65,
        step=0.05,
        help=(
            "Controls how strict the CEWS early-warning indicator is when flagging regime shifts in the CEWS tab. "
            "Raising the bar highlights only the strongest signals."
        ),
    )
    _sidebar_recommendation("0.65 catches material swings without flooding the dashboard with minor blips.")

    st.sidebar.markdown("### Robustness Stress")
    robustness_noise = st.sidebar.slider(
        "Noise injection (%)",
        min_value=0,
        max_value=15,
        value=5,
        step=1,
        help=(
            "Adds Gaussian noise to feature inputs before re-scoring risk to test stability. Higher values "
            "simulate data quality degradation and should only be used when probing robustness."
        ),
    )
    _sidebar_recommendation("5% noise reveals fragility without overwhelming the signal.")
    robustness_delay = st.sidebar.slider(
        "Reporting delay (days)",
        min_value=0,
        max_value=5,
        value=1,
        step=1,
        help=(
            "Shifts labels forward to mimic late reporting when re-running the models. Larger delays stress early "
            "warning ability but may break causal ordering for some studies."
        ),
    )
    _sidebar_recommendation("A 1-day delay approximates typical news update lags.")

    st.sidebar.markdown("### Glossary")
    with st.sidebar.expander("Key Terms", expanded=False):
        st.markdown(
            """
- **CEWS (Cross-modal Early Warning Score):** Measures lead time and reliability of the system's alerts, rewarding early, accurate warnings and penalizing false alarms.
- **Multimodal Fusion:** The process of aligning numeric market data with textual sentiment signals so models learn from a unified view.
- **Ensemble Learning:** Combines multiple base models, often with different inductive biases, to produce more stable and accurate risk forecasts.
- **SHAP (Shapley Additive Explanations):** Game-theoretic technique that attributes how each feature pushes a prediction higher or lower relative to a baseline.
- **GARCH (Generalized Autoregressive Conditional Heteroskedasticity):** A volatility model used as a classical benchmark for market risk that captures time-varying variance.
- **VaR (Value at Risk):** Statistical measure that estimates the maximum expected loss over a time horizon at a chosen confidence level.
"""
        )

    if "benchmark_results" not in st.session_state:
        st.session_state["benchmark_results"] = None
    if "case_study_results" not in st.session_state:
        st.session_state["case_study_results"] = None

    case_study_map = {scenario.name: scenario.slug for scenario in PREDEFINED_CASE_STUDIES}

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
        tab_pipeline,
        tab_case_studies,
        tab_model_results,
        tab_results_explorer,
    tab_experiment_tracker,
        tab_explainability,
        tab_multimodal,
        tab_feature_analysis,
        tab_risk_timeline,
        tab_sentiment,
        tab_data_explorer,
        tab_explanations,
        tab_model_details,
        tab_research,
    ) = st.tabs(
        [
            "🧪 Pipeline Lab",
            "📼 Case Studies",
            "📊 Model Results",
            "🗂️ Results Explorer",
            "🧾 Experiment Tracker",
            "🧠 Explainability",
            "🧬 Multimodal + CEWS",
            "📈 Feature Analysis",
            "📊 Risk Timeline",
            "📰 Sentiment Analysis",
            "🔍 Data Explorer",
            "❓ Explanations",
            "⚙️ Model Details",
            "📚 Research Addendum",
        ]
    )

    with tab_pipeline:
        st.subheader("Pipeline Orchestrator")
        st.write(
            "Configure the controls in the left sidebar, then trigger a live run to see stage-by-stage analytics based on your selections."
        )

        filtered_df = (
            df[df["Symbol"].isin(selected_symbols)].copy()
            if selected_symbols and "Symbol" in df.columns
            else df.copy()
        )

        if filtered_df.empty:
            st.warning("No data available for the chosen tickers. Expand your selection to run the pipeline.")
        elif not selected_stages:
            st.info("Select at least one pipeline stage in the sidebar to preview analytics.")
        else:
            sentiment_df = prepare_sentiment_dataframe(sentiment_choice, selected_symbols)
            robustness_config = {"noise": robustness_noise / 100.0, "delay": robustness_delay}
            run_summary = {
                "symbols": selected_symbols,
                "sentiment": sentiment_choice,
                "fusion": fusion_choice,
                "ensemble": ensemble_choice,
                "risk_threshold": risk_threshold,
                "cews_threshold": cews_threshold,
                "noise_percent": robustness_noise,
                "delay_days": robustness_delay,
            }

            if st.button("Run Pipeline with Current Settings", key="run_pipeline_button"):
                progress_bar = st.progress(0.0)
                progress_status = st.empty()
                progress_status.info("Initializing pipeline run…")

                def _on_stage_complete(stage_name: str, completed: int, total: int) -> None:
                    percent = completed / total if total else 1.0
                    progress_status.info(f"Completed {stage_name} ({completed}/{total})")
                    progress_bar.progress(min(1.0, percent))

                outputs = execute_stage_controls(
                    selected_stages,
                    selected_models,
                    fusion_choice,
                    ensemble_choice,
                    filtered_df,
                    results,
                    sentiment_df,
                    risk_threshold,
                    cews_threshold,
                    robustness_config,
                    progress_callback=_on_stage_complete,
                )

                st.session_state["pipeline_results"] = {
                    "stage_outputs": outputs,
                    "run_summary": run_summary,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

                case_selection = st.session_state.get("case_study_selection", [])
                if st.session_state.get("case_study_results") and case_selection:
                    st.session_state["case_studies_queue"] = list(case_selection)

                reset_streamlit_caches()
                st.session_state["pipeline_refresh_notice"] = True
                progress_bar.progress(1.0)
                progress_status.success("Pipeline run complete! Refreshing downstream tabs…")

                st.experimental_rerun()

            pipeline_results = st.session_state.get("pipeline_results")
            if pipeline_results and pipeline_results.get("stage_outputs"):
                if st.session_state.pop("pipeline_refresh_notice", False):
                    st.caption("✅ Downstream tabs refreshed with the latest pipeline artifacts.")

                summary = pipeline_results.get("run_summary", {})
                timestamp = pipeline_results.get("timestamp")
                subtitle_parts = []
                if summary.get("symbols"):
                    subtitle_parts.append(
                        f"Tickers: {', '.join(summary['symbols'])}"
                    )
                subtitle_parts.append(f"Fusion: {summary.get('fusion', fusion_choice).replace('_', ' ').title()}")
                subtitle_parts.append(f"Ensemble: {summary.get('ensemble', ensemble_choice)}")
                subtitle_parts.append(f"Risk Threshold: {summary.get('risk_threshold', risk_threshold):.2f}")
                subtitle = " • ".join(subtitle_parts)
                st.success(f"Last run {timestamp} — {subtitle}")
                render_stage_outputs(pipeline_results["stage_outputs"])
            else:
                st.info("Configure parameters and press the run button to see pipeline analytics.")

        st.markdown("---")
        st.subheader("Benchmarks & Case Studies")
        col_left, col_right = st.columns([1, 1])

        with col_left:
            if st.button("Run Benchmark Suite", key="run_benchmarks_button"):
                with st.spinner("Running baseline comparisons on a sampled dataset…"):
                    st.session_state["benchmark_results"] = run_benchmarks(filtered_df, ensemble_choice)

        with col_right:
            selected_case_names = st.multiselect(
                "Case studies to replay",
                options=list(case_study_map.keys()),
                help="Replay iconic crises with the latest cached predictions to inspect warning coverage.",
                key="case_study_selection",
            )

            queued_case_names = st.session_state.pop("case_studies_queue", None)
            if queued_case_names:
                valid_case_names = [name for name in queued_case_names if name in case_study_map]
                if valid_case_names:
                    with st.spinner("Refreshing case studies with latest results…"):
                        slugs = [case_study_map[name] for name in valid_case_names]
                        st.session_state["case_study_results"] = run_case_studies(slugs)

            if st.button("Run Case Studies", key="run_case_studies_button"):
                if not selected_case_names:
                    st.warning("Pick one or more case studies before running the replay.")
                else:
                    slugs = [case_study_map[name] for name in selected_case_names]
                    with st.spinner("Replaying selected scenarios…"):
                        st.session_state["case_study_results"] = run_case_studies(slugs)

        benchmark_container = st.container()
        if st.session_state.get("benchmark_results"):
            with benchmark_container:
                display_benchmark_results(st.session_state["benchmark_results"])

        case_study_container = st.container()
        case_results = st.session_state.get("case_study_results")
        if case_results:
            with case_study_container:
                display_case_study_results(case_results)

    with tab_case_studies:
        st.subheader("Crisis Replays")
        st.caption("Replay historical stress events to see how MEWS tracked rising risk versus actual market drawdowns.")

        scenario_whitelist = {"gfc_2008", "covid_crash", "fed_hikes"}
        scenario_options = [scenario for scenario in PREDEFINED_CASE_STUDIES if scenario.slug in scenario_whitelist]

        if not scenario_options:
            st.warning("No predefined case studies available. Check `src/case_studies/scenarios.py` for configurations.")
        else:
            scenario_map = {scenario.name: scenario for scenario in scenario_options}
            scenario_name = st.selectbox("Scenario", list(scenario_map.keys()))
            selected_scenario = scenario_map[scenario_name]

            st.markdown(f"_{selected_scenario.description}_")

            predictions_cache = load_predictions_cache()
            subset, timeline = prepare_case_study_dataset(df, selected_scenario, predictions_cache)

            if timeline.empty:
                st.info("No cached data available for this scenario. Run the full pipeline to populate integrated datasets and predictions.")
            else:
                min_date = timeline["Date"].min().to_pydatetime()
                max_date = timeline["Date"].max().to_pydatetime()

                if min_date == max_date:
                    selected_range = (min_date, max_date)
                else:
                    selected_range = st.slider(
                        "Timeline window",
                        min_value=min_date,
                        max_value=max_date,
                        value=(min_date, max_date),
                        format="YYYY-MM-DD",
                    )

                range_start = pd.Timestamp(selected_range[0])
                range_end = pd.Timestamp(selected_range[1])

                timeline_filtered = timeline[
                    (timeline["Date"] >= range_start) & (timeline["Date"] <= range_end)
                ].copy()
                subset_filtered = subset[
                    (subset["Date"] >= range_start) & (subset["Date"] <= range_end)
                ].copy()

                if timeline_filtered.empty:
                    st.info("Adjust the slider to select a window that contains data.")
                else:
                    peak_risk = float(timeline_filtered["Risk_Probability"].max())
                    deepest_drawdown = float(timeline_filtered["Drawdown"].min())
                    warning_rate = float(
                        (subset_filtered["Risk_Probability"] >= selected_scenario.risk_threshold).mean()
                        if not subset_filtered.empty
                        else 0.0
                    )

                    metric_cols = st.columns(3)
                    metric_cols[0].metric("Peak Risk Probability", f"{peak_risk:.2f}")
                    metric_cols[1].metric("Deepest Drawdown", f"{deepest_drawdown:.1%}")
                    metric_cols[2].metric(
                        "Warning Coverage",
                        f"{warning_rate:.0%}",
                        help="Share of days where risk probability exceeded the scenario threshold.",
                    )

                    case_fig = create_case_study_plot(timeline_filtered, selected_scenario, selected_scenario.risk_threshold)
                    if case_fig is not None:
                        case_fig.update_xaxes(range=[range_start, range_end])
                        _render_plotly(case_fig)
                        st.caption("Use the slider above to zoom into specific phases of the crisis timeline.")

                    preview_cols = [
                        col
                        for col in ["Date", "Symbol", "Risk_Probability", "Risk_Label", "Drawdown", "Risk_Prediction"]
                        if col in subset_filtered.columns
                    ]
                    with st.expander("Show sample data points"):
                        _render_dataframe(
                            subset_filtered[preview_cols].head(200),
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
                _render_plotly(confusion_fig)
        else:
            st.info(
                "Test prediction data unavailable. Run the training pipeline to populate evaluation metrics."
            )

        comparison_fig = create_model_comparison_chart(results)
        if comparison_fig:
            _render_plotly(
                comparison_fig,
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
            _render_dataframe(pd.DataFrame(rows))

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
                        pd.DataFrame(event_rows)
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

    with tab_results_explorer:
        st.subheader("Results Explorer")

        research_payload = results.get("research_artifacts") if isinstance(results, dict) else None
        if not research_payload:
            st.info(
                "Run the full pipeline from the CLI to populate experiment metrics, ablations, and robustness diagnostics."
            )
        else:
            evaluation = research_payload.get("evaluation_results", {}) or {}
            overall_metrics = evaluation.get("overall", {}) or {}
            calibration = evaluation.get("calibration", {}) or {}
            crisis_windows = evaluation.get("crisis_windows", {}) or {}
            hypothesis_results = research_payload.get("hypothesis_results", {}) or {}
            robustness_results = research_payload.get("robustness_results", {}) or {}

            def _fmt_metric(value: Optional[float]) -> str:
                if value is None:
                    return "N/A"
                try:
                    if pd.isna(value):
                        return "N/A"
                except TypeError:
                    pass

                try:
                    return f"{float(value):.3f}"
                except (TypeError, ValueError):
                    return str(value)

            if overall_metrics:
                metric_cols = st.columns(3)
                metric_cols[0].metric("AUC", _fmt_metric(overall_metrics.get("auc")))
                metric_cols[1].metric("Brier Score", _fmt_metric(overall_metrics.get("brier")))
                metric_cols[2].metric(
                    "Precision@50", _fmt_metric(overall_metrics.get("precision_at_k"))
                )

            st.markdown("#### Experiment Metrics")
            experiment_rows = []
            seen_experiments = set()
            if overall_metrics:
                experiment_rows.append(
                    {
                        "Experiment": "MEWS Ensemble",
                        "AUC": overall_metrics.get("auc"),
                        "Brier Score": overall_metrics.get("brier"),
                        "Precision@50": overall_metrics.get("precision_at_k"),
                        "Precision": overall_metrics.get("precision"),
                    }
                )
                seen_experiments.add("mews ensemble")

            experiments_payload = evaluation.get("experiments")
            if isinstance(experiments_payload, dict):
                for experiment_name, metrics in experiments_payload.items():
                    if not isinstance(metrics, dict):
                        continue
                    slug = experiment_name.strip().lower()
                    if slug in seen_experiments:
                        continue
                    experiment_rows.append(
                        {
                            "Experiment": experiment_name.replace("_", " ").title(),
                            "AUC": metrics.get("auc"),
                            "Brier Score": metrics.get("brier"),
                            "Precision@50": metrics.get("precision_at_k"),
                            "Precision": metrics.get("precision"),
                        }
                    )
                    seen_experiments.add(slug)

            if experiment_rows:
                experiments_download = pd.DataFrame(experiment_rows)
                experiments_display = experiments_download.copy()
                for col in [c for c in experiments_display.columns if c != "Experiment"]:
                    experiments_display[col] = experiments_display[col].apply(
                        lambda val: _fmt_metric(val)
                    )
                _render_dataframe(experiments_display)
                _render_download_buttons(
                    experiments_download,
                    "mews_experiment_metrics",
                    "results_explorer_experiments",
                )
            else:
                st.write("Experiment metrics unavailable.")

            st.markdown("#### Crisis Window Performance")
            if crisis_windows:
                crisis_download = (
                    pd.DataFrame(crisis_windows)
                    .T.reset_index()
                    .rename(columns={"index": "Crisis"})
                )
                crisis_download = crisis_download.rename(
                    columns=lambda col: col if col == "Crisis" else col.replace("_", " ").title()
                )
                crisis_display = crisis_download.copy()
                for col in [c for c in crisis_display.columns if c != "Crisis"]:
                    crisis_display[col] = crisis_display[col].apply(lambda val: _fmt_metric(val))
                _render_dataframe(crisis_display)
                _render_download_buttons(
                    crisis_download,
                    "mews_crisis_windows",
                    "results_explorer_crisis",
                )
            else:
                st.write("No crisis-specific evaluations recorded.")

            st.markdown("#### Ablation Studies")
            sentiment_result = hypothesis_results.get("sentiment_vs_fundamentals")
            graph_result = hypothesis_results.get("graph_feature_ablation")

            ablation_columns = st.columns(2)

            if sentiment_result:
                with ablation_columns[0]:
                    st.markdown("**Sentiment vs Fundamentals**")
                    sentiment_download = pd.DataFrame([sentiment_result]).rename(
                        columns={
                            "null_model_ll": "Base Log-Likelihood",
                            "alt_model_ll": "Full Log-Likelihood",
                            "lr_statistic": "Likelihood Ratio",
                            "p_value": "p-value",
                            "reject_null": "Reject Null?",
                        }
                    )
                    sentiment_display = sentiment_download.copy()
                    for col in ["Base Log-Likelihood", "Full Log-Likelihood", "Likelihood Ratio", "p-value"]:
                        if col in sentiment_display.columns:
                            sentiment_display[col] = sentiment_display[col].apply(
                                lambda val: _fmt_metric(val)
                            )
                    if "Reject Null?" in sentiment_display.columns:
                        sentiment_display["Reject Null?"] = sentiment_display["Reject Null?"].map(
                            {True: "Yes", False: "No"}
                        )
                    _render_dataframe(sentiment_display)
                    _render_download_buttons(
                        sentiment_download,
                        "mews_sentiment_ablation",
                        "results_explorer_sentiment",
                    )
            else:
                with ablation_columns[0]:
                    st.info("Sentiment ablation not available.")

            if graph_result:
                with ablation_columns[1]:
                    st.markdown("**Graph Feature Ablation**")
                    graph_download = pd.DataFrame([graph_result]).rename(
                        columns={
                            "auc_with_graph": "AUC (With Graph)",
                            "auc_without_graph": "AUC (Without Graph)",
                            "auc_difference": "AUC Difference",
                        }
                    )
                    graph_display = graph_download.copy()
                    for col in graph_display.columns:
                        graph_display[col] = graph_display[col].apply(lambda val: _fmt_metric(val))
                    _render_dataframe(graph_display)
                    _render_download_buttons(
                        graph_download,
                        "mews_graph_ablation",
                        "results_explorer_graph",
                    )
            else:
                with ablation_columns[1]:
                    st.info("Graph ablation not available.")

            st.markdown("#### Calibration Reliability Curve")
            true_probs = calibration.get("calibration_true") if calibration else None
            pred_probs = calibration.get("calibration_pred") if calibration else None
            if true_probs and pred_probs and len(true_probs) == len(pred_probs):
                cal_df = pd.DataFrame(
                    {
                        "Predicted Probability": pred_probs,
                        "Observed Frequency": true_probs,
                    }
                )
                cal_fig = go.Figure()
                cal_fig.add_trace(
                    go.Scatter(
                        x=cal_df["Predicted Probability"],
                        y=cal_df["Observed Frequency"],
                        mode="lines+markers",
                        name="Observed",
                        line=dict(color="#1f77b4", width=3),
                    )
                )
                cal_fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Perfect Calibration",
                        line=dict(color="#888888", dash="dash"),
                    )
                )
                cal_fig.update_layout(
                    xaxis_title="Predicted Probability",
                    yaxis_title="Observed Frequency",
                    height=350,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                _render_plotly(cal_fig)
                _render_download_buttons(
                    cal_df,
                    "mews_calibration_curve",
                    "results_explorer_calibration",
                )
            else:
                st.write("Calibration data unavailable.")

            st.markdown("#### Robustness Comparisons")
            bias_result = robustness_results.get("sentiment_bias") if robustness_results else None
            noise_result = robustness_results.get("adversarial_noise") if robustness_results else None

            if bias_result:
                bias_download = pd.DataFrame([bias_result]).rename(
                    columns={
                        "group_a": "Group A",
                        "group_b": "Group B",
                        "mean_a": "Mean Sentiment (A)",
                        "mean_b": "Mean Sentiment (B)",
                        "ks_statistic": "KS Statistic",
                        "p_value": "p-value",
                        "significant": "Significant?",
                    }
                )
                bias_display = bias_download.copy()
                for col in ["Mean Sentiment (A)", "Mean Sentiment (B)", "KS Statistic", "p-value"]:
                    if col in bias_display.columns:
                        bias_display[col] = bias_display[col].apply(lambda val: _fmt_metric(val))
                if "Significant?" in bias_display.columns:
                    bias_display["Significant?"] = bias_display["Significant?"].map({True: "Yes", False: "No"})
                st.markdown("**Sentiment Bias Diagnostics**")
                _render_dataframe(bias_display)
                _render_download_buttons(
                    bias_download,
                    "mews_sentiment_bias",
                    "results_explorer_bias",
                )

            if noise_result:
                noise_rows = []
                for scenario, metrics in noise_result.items():
                    if not isinstance(metrics, dict):
                        continue
                    row = {"Scenario": scenario.replace("_", " ").title()}
                    for metric_name, metric_value in metrics.items():
                        row[metric_name.replace("_", " ").title()] = metric_value
                    noise_rows.append(row)

                if noise_rows:
                    noise_download = pd.DataFrame(noise_rows)
                    noise_display = noise_download.copy()
                    for col in [c for c in noise_display.columns if c != "Scenario"]:
                        noise_display[col] = noise_display[col].apply(lambda val: _fmt_metric(val))
                    st.markdown("**Adversarial Noise Sensitivity**")
                    _render_dataframe(noise_display)

                    value_columns = [c for c in noise_download.columns if c != "Scenario"]
                    if value_columns:
                        noise_long = noise_download.melt(
                            id_vars="Scenario",
                            value_vars=value_columns,
                            var_name="Metric",
                            value_name="Score",
                        )
                        noise_fig = px.bar(
                            noise_long,
                            x="Metric",
                            y="Score",
                            color="Scenario",
                            barmode="group",
                        )
                        noise_fig.update_layout(height=360, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        _render_plotly(noise_fig)

                    _render_download_buttons(
                        noise_download,
                        "mews_robustness_noise",
                        "results_explorer_robustness",
                    )

            remaining_robustness = (
                {
                    key: value
                    for key, value in robustness_results.items()
                    if key not in {"sentiment_bias", "adversarial_noise"}
                }
                if robustness_results
                else {}
            )
            if remaining_robustness:
                st.markdown("**Additional Robustness Outputs**")
                st.json(remaining_robustness)

            st.markdown("#### Research Deliverables")
            md_path = research_payload.get("report_markdown")
            html_path = research_payload.get("report_html")

            if md_path and os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as handle:
                    st.download_button(
                        "Download Markdown Report",
                        handle.read(),
                        file_name=os.path.basename(md_path),
                        key="results_explorer_report_md",
                    )
            if html_path and os.path.exists(html_path):
                with open(html_path, "r", encoding="utf-8") as handle:
                    st.download_button(
                        "Download HTML Report",
                        handle.read(),
                        file_name=os.path.basename(html_path),
                        key="results_explorer_report_html",
                    )

    with tab_experiment_tracker:
        st.subheader("MLflow Experiment Tracker")

        if not MLFLOW_AVAILABLE:
            st.warning("MLflow isn't installed, so experiment tracking is disabled for this session.")
            st.markdown(
                "Install the optional dependency to enable this tab, then rerun the app after logging a pipeline run."
            )
            st.code("pip install mlflow", language="bash")
        else:
            client, experiment_id, mlflow_error = _init_mlflow_client()
            if mlflow_error:
                st.info(mlflow_error)
            else:
                runs_df, run_lookup, ordered_run_ids, runs_error = _load_mlflow_runs_table(
                    client, experiment_id
                )
                if runs_error:
                    st.error(runs_error)
                elif runs_df.empty:
                    st.info(
                        "No MLflow runs recorded yet for this experiment. Execute the training pipeline to populate history."
                    )
                else:
                    st.markdown("#### Recent Runs")
                    display_df = runs_df[[
                        "Run ID",
                        "Run",
                        "Status",
                        "Start",
                        "Duration (min)",
                        "Ensemble AUC",
                        "Brier",
                    ]].copy()
                    _render_dataframe(display_df)

                    run_label_map: Dict[str, str] = {}
                    option_labels: List[str] = []
                    for run_id in ordered_run_ids:
                        meta = run_lookup.get(run_id)
                        if not meta:
                            continue
                        start_label = (
                            meta["start"].strftime("%Y-%m-%d %H:%M")
                            if isinstance(meta.get("start"), pd.Timestamp)
                            else "N/A"
                        )
                        label = f"{meta['name']} • {start_label}"
                        option_labels.append(label)
                        run_label_map[label] = run_id

                    if not option_labels:
                        st.info("No completed run metadata is available yet.")
                    else:
                        selected_label = st.selectbox("Run for model card", option_labels, index=0)
                        selected_run_id = run_label_map[selected_label]
                        run_meta = run_lookup[selected_run_id]
                        run_results = _load_mlflow_json_artifact(
                            selected_run_id, "artifacts/model_results.json"
                        ) or {}
                        if not isinstance(run_results, dict):
                            run_results = {}

                        ensemble_metrics = run_results.get("ensemble", {}) if isinstance(run_results, dict) else {}

                        st.markdown("### Model Card")

                        start_time = run_meta.get("start")
                        end_time = run_meta.get("end")
                        start_display = (
                            start_time.strftime("%Y-%m-%d %H:%M")
                            if isinstance(start_time, pd.Timestamp)
                            else "—"
                        )
                        end_display = (
                            end_time.strftime("%Y-%m-%d %H:%M")
                            if isinstance(end_time, pd.Timestamp)
                            else "—"
                        )
                        duration_seconds = run_meta.get("duration_seconds")
                        duration_display = (
                            f"{duration_seconds / 60:.1f} min"
                            if isinstance(duration_seconds, (int, float))
                            else "—"
                        )
                        status_display = (
                            run_meta.get("status", "N/A").title()
                            if isinstance(run_meta.get("status"), str)
                            else str(run_meta.get("status", "N/A"))
                        )

                        metric_cols = st.columns(4)
                        metric_cols[0].metric(
                            "Ensemble AUC", _format_metric_value(ensemble_metrics.get("auc_score"))
                        )
                        metric_cols[1].metric(
                            "Accuracy", _format_metric_value(ensemble_metrics.get("test_accuracy"))
                        )
                        metric_cols[2].metric(
                            "Fβ Score", _format_metric_value(ensemble_metrics.get("fbeta_score"))
                        )
                        metric_cols[3].metric(
                            "Threshold", _format_metric_value(ensemble_metrics.get("optimal_threshold"))
                        )

                        meta_cols = st.columns(2)
                        with meta_cols[0]:
                            detail_lines = [
                                f"- **Run ID:** `{selected_run_id}`",
                                f"- **Status:** {status_display}",
                                f"- **Start:** {start_display}",
                                f"- **End:** {end_display}",
                                f"- **Duration:** {duration_display}",
                            ]
                            st.markdown("**Run Details**\n\n" + "\n".join(detail_lines))

                        with meta_cols[1]:
                            weights = ensemble_metrics.get("weights")
                            if isinstance(weights, dict) and weights:
                                weights_df = pd.DataFrame(
                                    [{"Model": key.upper(), "Weight": value} for key, value in weights.items()]
                                )
                                weights_df["Weight"] = weights_df["Weight"].apply(_format_metric_value)
                                st.markdown("**Ensemble Weights**")
                                _render_dataframe(weights_df)
                            else:
                                st.markdown("**Ensemble Weights**")
                                st.write("Weight allocations unavailable for this run.")

                        model_rows = []
                        for model_name, model_metrics in run_results.items():
                            if not isinstance(model_metrics, dict):
                                continue
                            if model_name in {"metadata", "test_data"}:
                                continue
                            if not any(
                                key in model_metrics for key in ("auc_score", "test_accuracy", "fbeta_score")
                            ):
                                continue
                            model_rows.append(
                                {
                                    "Model": model_name.replace("_", " ").title(),
                                    "AUC": model_metrics.get("auc_score"),
                                    "Accuracy": model_metrics.get("test_accuracy"),
                                    "Fβ": model_metrics.get("fbeta_score"),
                                }
                            )

                        if model_rows:
                            models_df = pd.DataFrame(model_rows)
                            models_display = models_df.copy()
                            for col in ["AUC", "Accuracy", "Fβ"]:
                                models_display[col] = models_display[col].apply(_format_metric_value)
                            st.markdown("**Per-model Metrics**")
                            _render_dataframe(models_display)

                        params = run_meta.get("params") or {}

                        with st.expander("MLflow Parameters", expanded=False):
                            if params:
                                params_df = pd.DataFrame(sorted(params.items()), columns=["Parameter", "Value"])
                                _render_dataframe(params_df)
                            else:
                                st.write("No parameters logged for this run.")

                        tags = run_meta.get("tags") or {}
                        if tags:
                            with st.expander("MLflow Tags", expanded=False):
                                tags_df = pd.DataFrame(sorted(tags.items()), columns=["Tag", "Value"])
                                _render_dataframe(tags_df)

                        st.markdown("### SHAP Feature Impact")
                        shap_candidates = [
                            name
                            for name, metrics in run_results.items()
                            if isinstance(metrics, dict) and name not in {"metadata", "test_data"}
                        ]
                        shap_label_map = {
                            name.replace("_", " ").title(): name for name in shap_candidates
                        }
                        if shap_label_map:
                            shap_selected_label = st.selectbox(
                                "Model for SHAP overview",
                                list(shap_label_map.keys()),
                                key=f"mlflow_shap_model_{selected_run_id}",
                            )
                            shap_model_key = shap_label_map[shap_selected_label]
                            with st.spinner("Computing SHAP summary..."):
                                shap_df = _compute_run_shap_global(selected_run_id, shap_model_key, df)
                            if shap_df is not None and not shap_df.empty:
                                top_shap = shap_df.head(20)
                                shap_fig = go.Figure(
                                    go.Bar(
                                        x=top_shap["importance"][::-1],
                                        y=top_shap["feature"][::-1],
                                        orientation="h",
                                        marker=dict(color="#7c3aed"),
                                    )
                                )
                                shap_fig.update_layout(
                                    title=f"Global SHAP Impact — {shap_selected_label}",
                                    height=480,
                                    margin=dict(l=160, r=40, t=60, b=40),
                                )
                                _render_plotly(shap_fig)
                            else:
                                st.info(
                                    "SHAP values unavailable. Ensure the SHAP library is installed and the chosen model artifact exists in MLflow."
                                )
                        else:
                            st.info("No model artifacts available for SHAP visualisation in this run.")

                        st.markdown("### Run Comparison")
                        default_compare = option_labels[: min(3, len(option_labels))]
                        selected_compares = st.multiselect(
                            "Select runs to compare",
                            option_labels,
                            default=default_compare,
                            help="Compare ensemble metrics across multiple MLflow runs.",
                        )
                        if selected_compares:
                            comparison_records = []
                            for compare_label in selected_compares:
                                run_id = run_label_map[compare_label]
                                compare_results = _load_mlflow_json_artifact(
                                    run_id, "artifacts/model_results.json"
                                )
                                ensemble = (
                                    compare_results.get("ensemble", {})
                                    if isinstance(compare_results, dict)
                                    else {}
                                )
                                comparison_records.append(
                                    {
                                        "Run": run_lookup[run_id]["name"],
                                        "AUC": ensemble.get("auc_score"),
                                        "Accuracy": ensemble.get("test_accuracy"),
                                        "Fβ": ensemble.get("fbeta_score"),
                                    }
                                )
                            comparison_df = pd.DataFrame(comparison_records)
                            if not comparison_df.empty:
                                comparison_display = comparison_df.copy()
                                for col in ["AUC", "Accuracy", "Fβ"]:
                                    comparison_display[col] = comparison_display[col].apply(
                                        _format_metric_value
                                    )
                                _render_dataframe(comparison_display)

                                melted = (
                                    comparison_df.melt(
                                        id_vars="Run",
                                        value_vars=["AUC", "Accuracy", "Fβ"],
                                        var_name="Metric",
                                        value_name="Value",
                                    )
                                    .dropna(subset=["Value"])
                                )
                                if not melted.empty:
                                    compare_fig = px.bar(
                                        melted,
                                        x="Run",
                                        y="Value",
                                        color="Metric",
                                        barmode="group",
                                    )
                                    compare_fig.update_layout(
                                        height=420,
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                    )
                                    _render_plotly(compare_fig)
                        else:
                            st.info("Select one or more runs to build a comparison chart.")

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
        sample_cap = min(len(feature_matrix), 1000)

        if model is None:
            st.warning(
                "No trained model artifacts found. Run the training pipeline to enable explainability insights."
            )
        elif feature_matrix.empty:
            st.warning("Feature matrix is empty; cannot compute explanations.")
        else:
            if sample_cap <= 10:
                sample_size = sample_cap
                st.caption(
                    "Using all available rows (≤10) for the global SHAP summary. Collect more data for richer explanations."
                )
            else:
                sample_min = min(sample_cap - 1, max(10, sample_cap // 2))
                sample_default = min(
                    sample_cap - 1,
                    max(sample_min, (sample_min + sample_cap) // 2),
                )
                slider_step = max(1, (sample_cap - sample_min) // 10)
                sample_size = st.slider(
                    "Sample size for global SHAP",
                    sample_min,
                    sample_cap,
                    sample_default,
                    step=slider_step,
                    help="Larger samples smooth the global importance chart; smaller samples react faster to new data.",
                )

            shap_input = (
                feature_matrix.sample(sample_size, random_state=42)
                if len(feature_matrix) > sample_size
                else feature_matrix
            )

            shap_global = None
            if not SHAP_AVAILABLE:
                st.info("Install the optional `shap` package to unlock global explanations (`pip install shap`).")
            else:
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
                _render_plotly(global_fig)
                st.markdown(
                    """
                    **How to read it:** Each bar shows the average absolute SHAP impact for a feature across the sampled rows. Longer bars mean the feature consistently nudged the model's risk score up or down. A flat bar implies the model rarely relied on that input. Look for clusters of related features (e.g., different volatility measures) to understand combined pressure on the prediction.

                    • Values are always positive because we take the average magnitude of the push.  
                    • If two features are close in height, they contributed nearly the same amount of influence.  
                    • Use this chart to spot which signals dominate the model so you can monitor or stress-test them directly.
                    """
                )
            elif SHAP_AVAILABLE:
                st.warning(
                    "Unable to compute SHAP values for the sampled data. Try reducing the sample size or rerunning the training pipeline."
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

            shap_local = None
            lime_explanation = None
            if SHAP_AVAILABLE or LIME_AVAILABLE:
                with st.spinner("Computing local SHAP and LIME explanations..."):
                    if SHAP_AVAILABLE:
                        shap_local = compute_local_shap_explanation(
                            model, feature_matrix, selected_position
                        )
                    if LIME_AVAILABLE:
                        lime_explanation = compute_lime_explanation(
                            model,
                            feature_matrix,
                            selected_position,
                            class_names=np.array(["Stable", "Risk"]),
                        )

            if shap_local is not None and not shap_local.empty:
                local_display = shap_local.nlargest(20, "abs_shap").iloc[::-1]
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
                _render_plotly(local_fig)
                st.markdown(
                    """
                    **Understanding the bars:** Positive (green) bars pushed the prediction toward **Risk**, while negative (red) bars pulled it toward **Stable**. The bar length equals that feature's SHAP value for the selected date/ticker. Compare these against the global chart to see whether today's drivers match the long-term leaders.

                    • SHAP values add up to the model's risk score once you include the baseline probability.  
                    • Large positive + large negative bars can offset each other; the mix tells you whether the day was borderline or clearly risky.  
                    • Hover each bar to see the exact contribution in probability points.
                    """
                )
            elif SHAP_AVAILABLE:
                st.info("Local SHAP explanations unavailable for the selected sample.")
            else:
                st.info("Install the optional `shap` package to enable local explanations (`pip install shap`).")

            if lime_explanation is not None and not lime_explanation.empty:
                lime_display = lime_explanation.nlargest(20, "abs_weight").iloc[::-1]
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
                _render_plotly(lime_fig)
                st.caption(
                    "LIME perturbs the original row to learn a tiny linear model around it. Positive weights (green) argue for the risk class, negative weights (red) argue for the stable class. Compare them with the SHAP bars: if both methods agree on the top signals, the explanation is more trustworthy."
                )
            elif LIME_AVAILABLE:
                st.info("LIME explanation unavailable for the selected sample.")
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
                _render_dataframe(df[subset].head(5))
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
            _render_plotly(timeline_fig)

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
            _render_plotly(importance_fig, config=config)

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

            _render_plotly(fig)

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
            _render_plotly(timeline_fig, config=config)

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
                _render_plotly(fig)

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
                                _render_plotly(sentiment_chart)

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

                            _render_dataframe(symbol_sentiment)

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
        _render_dataframe(df.head(100))

        # Data statistics
        st.subheader("Statistical Summary")
        _render_dataframe(df.describe())

    with tab_explanations:
        st.subheader("Concept Guide")
        st.caption(
            "Collapsible explainers that translate MEWS concepts into plain language."
        )

        with st.expander("What is MEWS?", expanded=False):
            st.markdown(
                """
                **MEWS** stands for *Market Risk Early Warning System*. It combines several data feeds so we can spot rising market stress early instead of reacting late.

                ```text
                Market Prices ┐             ┌─> Machine-Learning Models ──> Risk Alerts
                               ├─> Cleaning ─┤
                News & Filings ┘             └─> Research Reports & Dashboards
                ```

                **Key idea:** by monitoring prices *and* language in news/filings, MEWS hears the warning whispers before the crash.
                """
            )

        with st.expander("What is CEWS (cross-modal early warning extension)?", expanded=False):
            st.markdown(
                """
                **CEWS** measures how timely and trustworthy the alerts are. It rewards warnings that fire before a downturn and penalises false alarms.

                ```text
                Timeline ───►  [ Alert ]====(lead time)====[ Downturn ]
                                  ▲             ▲
                                  │             └─ Reward: early, correct warning
                                  └─ Penalty if alert happens without any drop
                ```

                A higher CEWS score means "MEWS usually speaks up early and rarely cries wolf."
                """
            )

        with st.expander("What is multimodal fusion?", expanded=False):
            st.markdown(
                """
                **Multimodal fusion** is the glue that lines up numbers and words so the model sees one coherent table.

                ```text
                ┌────────┐   ┌──────────────┐   ┌────────────┐
                │ Prices │ + │ News sentiment│ + │ SEC signals│
                └────────┘   └──────────────┘   └────────────┘
                           ↓ align by ticker & date ↓
                        ┌───────────────────────────┐
                        │ Unified feature matrix     │
                        └───────────────────────────┘
                ```

                Once fused, the model can learn patterns that exist only when you look across all data types together.
                """
            )

        with st.expander("What is ensemble learning & regime adaptation?", expanded=False):
            st.markdown(
                """
                **Ensemble learning** mixes multiple models so their strengths add up. **Regime adaptation** changes those weights when the market mood flips.

                ```text
                Calm market:     60% Logistic  + 40% RandomForest
                Choppy market:   20% Logistic  + 40% GradientBoost + 40% XGBoost
                ```

                This way no single model has to be perfect everywhere—MEWS leans on whichever model works best for the current regime.
                """
            )

        with st.expander("How does SHAP work?", expanded=False):
            st.markdown(
                """
                **SHAP** explains a prediction by showing how each feature pushed the risk score up or down compared with a neutral baseline.

                ```text
                Base risk = 0.40
                + Volatility spike        (+0.18)
                + Negative news headline  (+0.10)
                - Strong liquidity        (-0.05)
                ---------------------------------
                Final risk = 0.63
                ```

                Think of SHAP as a receipt: every line item tells you why the model believed risk was high (or low) for that date and ticker.
                """
            )

        with st.expander("What do the robustness tests mean?", expanded=False):
            st.markdown(
                """
                **Robustness tests** shake the data slightly—adding noise or delaying labels—to see if MEWS still raises similar alerts.

                ```text
                Original data ──► Predict ──► Baseline metrics
                   │
                   ├─ add noise (+5%) ──► Predict ──► Compare change?
                   └─ shift labels (+1 day) ──► Predict ──► Still alert on time?
                ```

                If the metrics barely move, the system is sturdy. Big drops tell us where to harden the pipeline.
                """
            )

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
                    _render_dataframe(report_df)

                st.divider()

    with tab_research:
        st.subheader("Research Addendum")

        research_payload = results.get("research_artifacts") if isinstance(results, dict) else None
        if not research_payload:
            st.info("Research artifacts will appear here after running the full pipeline on the CLI.")
        else:
            st.success("Research artifacts generated from the latest pipeline run.")

            evaluation = research_payload.get("evaluation_results", {})
            overall_metrics = evaluation.get("overall", {}) if evaluation else {}
            calibration = evaluation.get("calibration", {}) if evaluation else {}
            crisis_windows = evaluation.get("crisis_windows", {}) if evaluation else {}

            summary_cols = st.columns(3)
            summary_cols[0].metric(
                "AUC",
                f"{overall_metrics.get('auc', float('nan')):.3f}" if overall_metrics else "N/A",
            )
            summary_cols[1].metric(
                "Brier Score",
                f"{overall_metrics.get('brier', float('nan')):.3f}" if overall_metrics else "N/A",
            )
            summary_cols[2].metric(
                "Precision@50",
                f"{overall_metrics.get('precision_at_k', float('nan')):.3f}"
                if overall_metrics
                else "N/A",
            )

            st.markdown("#### Evaluation Metrics Overview")
            if overall_metrics:
                metrics_df = pd.DataFrame([overall_metrics]).rename(
                    columns=lambda col: col.replace("_", " ").title()
                )
                _render_dataframe(metrics_df)
            else:
                st.write("Evaluation metrics unavailable.")

            st.markdown("#### Calibration Reliability Curve")
            true_probs = calibration.get("calibration_true") if calibration else None
            pred_probs = calibration.get("calibration_pred") if calibration else None
            if true_probs and pred_probs and len(true_probs) == len(pred_probs):
                cal_fig = go.Figure()
                cal_fig.add_trace(
                    go.Scatter(
                        x=pred_probs,
                        y=true_probs,
                        mode="lines+markers",
                        name="Observed",
                        line=dict(color="#1f77b4", width=3),
                    )
                )
                cal_fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Perfect Calibration",
                        line=dict(color="#888888", dash="dash"),
                    )
                )
                cal_fig.update_layout(
                    xaxis_title="Predicted Probability",
                    yaxis_title="Observed Frequency",
                    height=350,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                _render_plotly(cal_fig)
            else:
                st.write("Calibration data unavailable.")

            st.markdown("#### Crisis Period Benchmarks")
            if crisis_windows:
                crisis_df = (
                    pd.DataFrame(crisis_windows)
                    .T.reset_index()
                    .rename(columns={"index": "Crisis"})
                )
                crisis_df = crisis_df.rename(
                    columns=lambda col: col if col == "Crisis" else col.replace("_", " ").title()
                )
                _render_dataframe(crisis_df)
            else:
                st.write("No crisis window evaluations recorded.")

            st.markdown("### Hypothesis Tests")
            hypothesis_results = research_payload.get("hypothesis_results", {})
            if hypothesis_results:
                hypothesis_df = (
                    pd.DataFrame(hypothesis_results)
                    .T.reset_index()
                    .rename(columns={"index": "Test"})
                )
                hypothesis_df = hypothesis_df.rename(
                    columns=lambda col: col if col == "Test" else col.replace("_", " ").title()
                )
                _render_dataframe(hypothesis_df)
            else:
                st.write("No hypothesis tests recorded.")

            st.markdown("### Robustness Checks")
            robustness_results = research_payload.get("robustness_results", {})
            if robustness_results:
                bias_result = robustness_results.get("sentiment_bias")
                if bias_result:
                    st.markdown("**Sentiment Bias Diagnostics**")
                    bias_df = pd.DataFrame([bias_result]).rename(
                        columns=lambda col: col.replace("_", " ").title()
                    )
                    _render_dataframe(bias_df)

                noise_result = robustness_results.get("adversarial_noise")
                if noise_result:
                    st.markdown("**Adversarial Noise Sensitivity**")
                    noise_df = (
                        pd.DataFrame(noise_result)
                        .T.reset_index()
                        .rename(columns={"index": "Scenario"})
                    )
                    noise_df = noise_df.rename(
                        columns=lambda col: col if col == "Scenario" else col.replace("_", " ").title()
                    )
                    _render_dataframe(noise_df)

                remaining_keys = {
                    key: value
                    for key, value in robustness_results.items()
                    if key not in {"sentiment_bias", "adversarial_noise"}
                }
                if remaining_keys:
                    st.json(remaining_keys)
            else:
                st.write("No robustness diagnostics recorded.")

            st.markdown("### Unique & Novel Contributions")
            st.markdown(
                """
- **Regime-Adaptive Ensemble:** Dynamically reweights classical ML models by volatility regime to capture market regime shifts.
- **Cross-Modal Attention Fusion:** Aligns textual sentiment embeddings with tabular indicators for richer early-warning signals.
- **Research Benchmarking Suite:** Tracks MEWS against GARCH/VaR and deep LSTM baselines with crisis-specific scorecards.
- **Causal & Ethical Diagnostics:** Automates likelihood-ratio tests, graph ablations, and sentiment bias checks to validate signal integrity.
- **Auto-Generated Research Reports:** Publishes Markdown and HTML dossiers ready for peer review in `outputs/research/`.
"""
            )

            st.markdown("### Download Reports")
            md_path = research_payload.get("report_markdown")
            html_path = research_payload.get("report_html")
            if md_path and os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as handle:
                    st.download_button(
                        "Download Markdown Report",
                        handle.read(),
                        file_name=os.path.basename(md_path),
                    )
            if html_path and os.path.exists(html_path):
                with open(html_path, "r", encoding="utf-8") as handle:
                    st.download_button(
                        "Download HTML Report",
                        handle.read(),
                        file_name=os.path.basename(html_path),
                    )


if __name__ == "__main__":
    main()
