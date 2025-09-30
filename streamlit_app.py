"""
Streamlit Frontend for Market Risk Early Warning System
Interactive dashboard with 4 ML models comparison
"""

import json
import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Import our system components
import sys

sys.path.append(".")
from datetime import datetime, timedelta

import requests

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
        st.info("✅ Loaded dataset with pre-computed risk labels")
    elif os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
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
        news_df["published_date"] = pd.to_datetime(news_df["published_date"])
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
                stock_data["Date"] = pd.to_datetime(stock_data["Date"])
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

    # Get feature importance from Random Forest (most interpretable)
    rf_results = results["random_forest"]
    if "feature_importance" not in rf_results:
        return None

    importance = rf_results["feature_importance"]
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]

    # Create user-friendly names
    feature_names = []
    for feature in sorted_features:
        original_name = feature[0]
        # Convert technical names to user-friendly names
        friendly_name = original_name

        # Technical -> User-friendly mappings
        name_mappings = {
            "rsi": "RSI (Price Momentum)",
            "macd": "MACD (Trend Indicator)",
            "bollinger": "Bollinger Bands (Volatility)",
            "ma_": "Moving Average ",
            "volume": "Trading Volume",
            "close": "Stock Price",
            "high": "Daily High Price",
            "low": "Daily Low Price",
            "open": "Opening Price",
            "volatility": "Price Volatility",
            "return": "Daily Return",
            "sentiment": "News Sentiment",
            "correlation": "Market Correlation",
            "drawdown": "Price Drawdown",
            "sharpe": "Risk-Return Ratio",
            "vix": "Market Fear Index (VIX)",
        }

        for tech, friendly in name_mappings.items():
            if tech.lower() in original_name.lower():
                friendly_name = friendly
                break

        # Clean up remaining underscores and make title case
        if friendly_name == original_name:
            friendly_name = original_name.replace("_", " ").title()

        feature_names.append(friendly_name)

    features, scores = zip(*sorted_features)

    # Create color gradient based on importance
    colors = [
        (
            "rgba(255,99,71,0.8)"
            if score > 0.1
            else "rgba(255,165,0,0.8)" if score > 0.05 else "rgba(135,206,235,0.8)"
        )
        for score in scores
    ]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=feature_names,
            orientation="h",
            marker_color=colors,
            text=[f"{score:.3f}" for score in scores],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="🎯 Most Important Factors for Risk Prediction",
        xaxis_title="Importance Level (0 = Not Important, 1 = Very Important)",
        yaxis_title="Market Factors",
        height=500,
        showlegend=False,
        font=dict(size=12),
    )

    return fig


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
            timeline_data["Date"] = pd.to_datetime(timeline_data["Date"])
        except:
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

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "🎯 Model Performance",
            "📈 Feature Analysis",
            "📊 Risk Timeline",
            "📰 Sentiment Analysis",
            "🔍 Data Explorer",
            "⚙️ Model Details",
        ]
    )

    with tab1:
        st.subheader("ML Model Performance Comparison")

        col1, col2, col3, col4 = st.columns(4)

        # Display model metrics
        model_count = 0
        total_accuracy = 0
        best_model = ""
        best_auc = 0

        for model_name, metrics in results.items():
            if model_name not in ["test_data", "ensemble"] and isinstance(
                metrics, dict
            ):
                model_count += 1
                accuracy = metrics.get("test_accuracy", 0)
                auc = metrics.get("auc_score", 0)
                total_accuracy += accuracy

                if auc > best_auc:
                    best_auc = auc
                    best_model = model_name.replace("_", " ").title()

        with col1:
            st.metric("Models Trained", model_count)
        with col2:
            st.metric("Avg Accuracy", f"{total_accuracy/max(model_count, 1):.3f}")
        with col3:
            st.metric("Best Model", best_model)
        with col4:
            st.metric("Best AUC", f"{best_auc:.3f}")

        # Model comparison chart
        comparison_fig = create_model_comparison_chart(results)
        if comparison_fig:
            st.plotly_chart(comparison_fig, width="stretch")

        # Model details table
        st.subheader("Detailed Model Results")

        model_data = []
        for model_name, metrics in results.items():
            if model_name not in ["test_data"] and isinstance(metrics, dict):
                model_data.append(
                    {
                        "Model": model_name.replace("_", " ").title(),
                        "Accuracy": f"{metrics.get('test_accuracy', 0):.4f}",
                        "AUC Score": f"{metrics.get('auc_score', 0):.4f}",
                        "Training Time": "N/A",  # Add if available
                    }
                )

        if model_data:
            st.dataframe(pd.DataFrame(model_data), width="stretch")

    with tab2:
        st.subheader("📈 What Drives Market Risk? (Plain English)")

        st.info(
            "💡 This shows which market factors are most important for predicting risk. Think of it like a recipe - these are the main ingredients that create market risk!"
        )

        # Feature importance chart
        importance_fig = create_feature_importance_chart(results)
        if importance_fig:
            st.plotly_chart(importance_fig, width="stretch")

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

            st.plotly_chart(fig, width="stretch")

            with st.expander("❓ How to Read This Chart"):
                st.write("• **Dark Red (+1.0)**: Factors move perfectly together")
                st.write("• **White (0.0)**: No relationship between factors")
                st.write("• **Dark Blue (-1.0)**: Factors move in opposite directions")
                st.write("• **Light colors**: Weak relationships")
                st.write(
                    "• **Tip**: Look for clusters of red or blue to find factor groups"
                )

    with tab3:
        st.subheader("📊 Market Risk Timeline - Your Risk Radar")

        st.info(
            "💡 **What am I looking at?** This shows how risky the market was on different days. Think of it like a weather forecast - but for your investments!"
        )

        # Risk timeline
        timeline_fig = create_risk_timeline(df)
        if timeline_fig:
            st.plotly_chart(timeline_fig, width="stretch")

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
                        available_symbols = df['Symbol'].unique() if 'Symbol' in df.columns else []
                        selected_symbols = st.sidebar.multiselect(
                            "Select symbols for enhanced analysis:", 
                            available_symbols, 
                            default=available_symbols[:3] if len(available_symbols) >= 3 else available_symbols
                        )
                        
                        if selected_symbols:
                            # Run enhanced analysis
                            enhanced_results = integrate_enhanced_risk_system(df, selected_symbols)
                            
                            if 'enhanced_dataset' in enhanced_results:
                                st.success(f"✅ Enhanced analysis complete! Added {enhanced_results.get('feature_count', 0)} new features")
                                
                                # Show enhanced timeline link
                                if enhanced_results.get('timeline_path'):
                                    st.markdown(f"📊 **Enhanced Interactive Timeline**: [View Enhanced Dashboard]({enhanced_results['timeline_path']})")
                                
                                # Show summary dashboard link  
                                if enhanced_results.get('summary_dashboard_path'):
                                    st.markdown(f"📋 **Risk Summary Dashboard**: [View Summary]({enhanced_results['summary_dashboard_path']})")
                                
                                # Show key metrics
                                enhanced_df = enhanced_results['enhanced_dataset']
                                if 'composite_risk_score' in enhanced_df.columns:
                                    avg_risk = enhanced_df['composite_risk_score'].mean()
                                    max_risk = enhanced_df['composite_risk_score'].max()
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Average Risk Score", f"{avg_risk:.3f}")
                                    with col2:
                                        st.metric("Maximum Risk Score", f"{max_risk:.3f}")
                                    with col3:
                                        risk_level = "🟢 LOW" if avg_risk < 0.3 else "🟡 MEDIUM" if avg_risk < 0.7 else "🔴 HIGH"
                                        st.metric("Risk Level", risk_level)
                            else:
                                st.error("❌ Enhanced analysis failed. Check the logs for details.")
                        else:
                            st.warning("Please select at least one symbol for enhanced analysis.")
                            
                    except Exception as e:
                        st.error(f"❌ Enhanced analysis error: {str(e)}")
            
            st.info("💡 **Enhanced Features Include**: Advanced sentiment analysis, market regime detection, volatility scoring, and composite risk indicators!")
        
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
                st.plotly_chart(fig, width="stretch")

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
            df_recent = df.copy()
            df_recent["Date"] = pd.to_datetime(df_recent["Date"])
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

    with tab4:
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
                                st.plotly_chart(sentiment_chart, width="stretch")

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

                            st.dataframe(symbol_sentiment, width="stretch")

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

    with tab5:
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
                    date_col = pd.to_datetime(df["Date"])
                    date_range = date_col.dt.date
                    st.metric("Date Range", f"{date_range.min()} to {date_range.max()}")
                except:
                    st.metric("Date Range", "Available")

        # Data preview
        st.subheader("Dataset Preview")
        st.dataframe(df.head(100), width="stretch")

        # Data statistics
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), width="stretch")

    with tab6:
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
                    st.dataframe(report_df, width="stretch")

                st.divider()


if __name__ == "__main__":
    main()
