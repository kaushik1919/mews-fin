"""
Visualization dashboard for market risk analysis
Creates interactive charts and static plots using matplotlib and Plotly
"""

import json
import logging
import os
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from .metrics import CEWSResult, compute_cews_score
from .xai_utils import (
    compute_global_shap_importance,
    compute_lime_explanation,
    compute_local_shap_explanation,
)


class RiskVisualizer:
    """Creates visualizations for market risk analysis"""

    def __init__(self, output_dir: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir or "outputs/visualizations"
        os.makedirs(self.output_dir, exist_ok=True)

        # Try to import visualization libraries
        self.matplotlib_available = self._check_matplotlib()
        self.plotly_available = self._check_plotly()

    def _check_matplotlib(self) -> bool:
        """Check if matplotlib is available"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            return True
        except ImportError:
            self.logger.warning(
                "Matplotlib/Seaborn not available - static plots disabled"
            )
            return False

    def _check_plotly(self) -> bool:
        """Check if plotly is available"""
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            return True
        except ImportError:
            self.logger.warning("Plotly not available - interactive plots disabled")
            return False

    def create_risk_dashboard(
        self,
        df: pd.DataFrame,
        symbols: Optional[List[str]] = None,
        predictions_df: Optional[pd.DataFrame] = None,
        model: Optional[Any] = None,
        feature_df: Optional[pd.DataFrame] = None,
        explainability_sample: int = 0,
        class_names: Optional[np.ndarray] = None,
    ) -> Dict[str, str]:
        """
        Create comprehensive risk dashboard

        Args:
            df: Integrated dataset with risk features
            symbols: List of symbols to visualize (if None, use all)
            predictions_df: DataFrame with model predictions
            model: Trained estimator used for SHAP/LIME explainability
            feature_df: Feature matrix aligned with the trained estimator
            explainability_sample: Row index used for local explanations
            class_names: Optional class labels passed to LIME for readability

        Returns:
            Dictionary with paths to generated visualizations and explainability assets
        """
        self.logger.info("Creating risk dashboard...")

        if df.empty:
            self.logger.error("No data provided for visualization")
            return {}

        # Select symbols to visualize
        if symbols is None:
            symbols = df["Symbol"].unique()[:10] if "Symbol" in df.columns else []
        else:
            symbols = list(symbols)

        enriched_df = self._attach_external_sentiment(df, symbols)

        visualization_paths = {}

        # 1. Stock price and risk timeline
        if self.plotly_available:
            path = self.plot_risk_timeline_interactive(
                enriched_df, symbols, predictions_df
            )
            if path:
                visualization_paths["risk_timeline_interactive"] = path

        if self.matplotlib_available:
            path = self.plot_risk_timeline_static(enriched_df, symbols, predictions_df)
            if path:
                visualization_paths["risk_timeline_static"] = path

        # 2. Sentiment analysis visualizations
        if self.matplotlib_available:
            path = self.plot_sentiment_analysis(enriched_df, symbols)
            if path:
                visualization_paths["sentiment_analysis"] = path

        # 3. Feature importance plots
        if self.matplotlib_available:
            path = self.plot_feature_importance(enriched_df)
            if path:
                visualization_paths["feature_importance"] = path

        # 4. Risk correlation heatmap
        if self.matplotlib_available:
            path = self.plot_risk_correlations(enriched_df)
            if path:
                visualization_paths["risk_correlations"] = path

        # 5. Model performance visualizations
        if predictions_df is not None and self.matplotlib_available:
            path = self.plot_model_performance(predictions_df)
            if path:
                visualization_paths["model_performance"] = path

        # 6. Market regime analysis
        if self.plotly_available:
            path = self.plot_market_regimes(enriched_df)
            if path:
                visualization_paths["market_regimes"] = path

        if model is not None and feature_df is not None:
            explainability_paths = self.create_explainability_dashboard(
                model,
                feature_df,
                sample_index=explainability_sample,
                class_names=class_names,
            )
            visualization_paths.update(explainability_paths)

        if predictions_df is not None and not predictions_df.empty:
            cews_paths = self.create_cews_dashboard(predictions_df)
            visualization_paths.update(cews_paths)

        self.logger.info(f"Created {len(visualization_paths)} visualizations")
        return visualization_paths

    def _attach_external_sentiment(
        self, df: pd.DataFrame, symbols: List[str]
    ) -> pd.DataFrame:
        """Ensure news sentiment scores are available for visualization outputs."""

        if df.empty or "Date" not in df.columns:
            return df

        if "news_sentiment_mean" in df.columns and df["news_sentiment_mean"].notna().any():
            return df

        sentiment_path = os.path.join(
            self.output_dir, "news_sentiment_timeseries.csv"
        )
        if not os.path.exists(sentiment_path):
            self.logger.debug(
                "Sentiment timeseries not found at %s; skipping sentiment enrichment",
                sentiment_path,
            )
            return df

        try:
            sentiment_df = pd.read_csv(sentiment_path)
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.warning(
                "Failed to load sentiment timeseries from %s: %s",
                sentiment_path,
                exc,
            )
            return df

        if sentiment_df.empty or "Date" not in sentiment_df.columns:
            return df

        sentiment_df["Date"] = pd.to_datetime(
            sentiment_df["Date"], errors="coerce"
        )
        sentiment_df = sentiment_df.dropna(subset=["Date"])

        if "Symbol" in sentiment_df.columns and symbols:
            sentiment_df = sentiment_df[sentiment_df["Symbol"].isin(symbols)]

        if sentiment_df.empty:
            return df

        merged_df = df.copy()
        merged_df["Date"] = pd.to_datetime(merged_df["Date"], errors="coerce")
        merged_df = merged_df.dropna(subset=["Date"])

        merge_value_map = {
            "Combined_Sentiment": "external_combined_sentiment",
            "News_Sentiment": "external_news_sentiment",
            "Risk_Probability": "external_sentiment_risk_probability",
        }
        available_cols = [col for col in merge_value_map if col in sentiment_df.columns]
        if not available_cols:
            return merged_df

        join_cols = ["Date"]
        use_symbol = False
        if "Symbol" in sentiment_df.columns and "Symbol" in merged_df.columns:
            join_cols.insert(0, "Symbol")
            use_symbol = True

        sentiment_prepped = sentiment_df[join_cols + available_cols].rename(
            columns={col: merge_value_map[col] for col in available_cols}
        )

        if use_symbol:
            merged_df = merged_df.dropna(subset=["Symbol"])
            sentiment_prepped = sentiment_prepped.dropna(subset=["Symbol"])

        merged_df["__original_index"] = np.arange(len(merged_df))

        tolerance = pd.Timedelta(days=5 * 365)

        try:
            if use_symbol:
                enriched_parts = []
                for symbol in merged_df["Symbol"].unique():
                    base = merged_df[merged_df["Symbol"] == symbol].sort_values(
                        "Date", kind="mergesort"
                    )
                    external = sentiment_prepped[
                        sentiment_prepped["Symbol"] == symbol
                    ].sort_values("Date", kind="mergesort")

                    if external.empty:
                        enriched_parts.append(base)
                        continue

                    merged_symbol = pd.merge_asof(
                        base,
                        external.drop(columns=["Symbol"]),
                        on="Date",
                        direction="nearest",
                        tolerance=tolerance,
                    )
                    enriched_parts.append(merged_symbol)

                enriched = (
                    pd.concat(enriched_parts, ignore_index=True)
                    if enriched_parts
                    else merged_df.copy()
                )
            else:
                base_sorted = merged_df.sort_values("Date", kind="mergesort")
                external_sorted = sentiment_prepped.sort_values(
                    "Date", kind="mergesort"
                )
                enriched = pd.merge_asof(
                    base_sorted,
                    external_sorted,
                    on="Date",
                    direction="nearest",
                    tolerance=tolerance,
                )
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.warning(
                "Failed to align sentiment timeseries with integrated data: %s", exc
            )
            merged_df.drop(columns=["__original_index"], inplace=True)
            return merged_df

        enriched = enriched.sort_values("__original_index", kind="mergesort")
        enriched = enriched.drop(columns=["__original_index"])

        candidate_sources = []
        if "external_combined_sentiment" in enriched.columns:
            candidate_sources.append(enriched["external_combined_sentiment"])
        if "external_news_sentiment" in enriched.columns:
            candidate_sources.append(enriched["external_news_sentiment"])

        if candidate_sources:
            if "news_sentiment_mean" in enriched.columns:
                for candidate in candidate_sources:
                    enriched["news_sentiment_mean"] = enriched["news_sentiment_mean"].fillna(
                        candidate
                    )
            else:
                enriched["news_sentiment_mean"] = candidate_sources[0]

        if (
            "news_sentiment_mean" in enriched.columns
            and "Symbol" in enriched.columns
            and enriched["news_sentiment_mean"].notna().any()
        ):
            enriched["news_sentiment_mean"] = (
                enriched.groupby("Symbol")["news_sentiment_mean"]
                .transform(lambda s: s.fillna(method="ffill").fillna(method="bfill"))
            )

        if "external_sentiment_risk_probability" in enriched.columns:
            if "Risk_Probability" in enriched.columns:
                enriched["Risk_Probability"] = enriched["Risk_Probability"].fillna(
                    enriched["external_sentiment_risk_probability"]
                )
            else:
                enriched["Risk_Probability"] = enriched[
                    "external_sentiment_risk_probability"
                ]

        columns_to_drop = [
            col
            for col in [
                "external_combined_sentiment",
                "external_news_sentiment",
                "external_sentiment_risk_probability",
            ]
            if col in enriched.columns
        ]
        if columns_to_drop:
            enriched.drop(columns=columns_to_drop, inplace=True)

        return enriched

    def plot_risk_timeline_interactive(
        self,
        df: pd.DataFrame,
        symbols: List[str],
        predictions_df: Optional[pd.DataFrame] = None,
    ) -> Optional[str]:
        """Create interactive risk timeline using Plotly"""

        if not self.plotly_available or df.empty:
            return None

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Select data for chosen symbols
            symbol_data = (
                df[df["Symbol"].isin(symbols)].copy()
                if "Symbol" in df.columns
                else df.copy()
            )

            if symbol_data.empty:
                return None

            if "Date" in symbol_data.columns:
                symbol_data["Date"] = pd.to_datetime(
                    symbol_data["Date"], errors="coerce"
                )
                symbol_data = symbol_data.dropna(subset=["Date"])
                symbol_data = symbol_data.sort_values("Date")

            # Create subplots
            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                subplot_titles=[
                    "Stock Prices & Risk Signals",
                    "Sentiment Scores",
                    "Risk Predictions",
                ],
                vertical_spacing=0.05,
                row_heights=[0.5, 0.25, 0.25],
            )

            colors = [
                "blue",
                "red",
                "green",
                "orange",
                "purple",
                "brown",
                "pink",
                "gray",
                "olive",
                "cyan",
            ]

            for i, symbol in enumerate(
                symbols[:10]
            ):  # Limit to 10 symbols for readability
                symbol_df = symbol_data[symbol_data["Symbol"] == symbol]
                if symbol_df.empty:
                    continue

                color = colors[i % len(colors)]

                # Plot 1: Stock prices
                if "Close" in symbol_df.columns and "Date" in symbol_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=symbol_df["Date"],
                            y=symbol_df["Close"],
                            mode="lines",
                            name=f"{symbol} Price",
                            line=dict(color=color),
                            yaxis="y1",
                        ),
                        row=1,
                        col=1,
                    )

                    # Add risk events as markers
                    if "Risk_Label" in symbol_df.columns:
                        risk_events = symbol_df[symbol_df["Risk_Label"] == 1]
                        if not risk_events.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=risk_events["Date"],
                                    y=risk_events["Close"],
                                    mode="markers",
                                    name=f"{symbol} Risk Events",
                                    marker=dict(
                                        color="red", size=8, symbol="triangle-up"
                                    ),
                                    yaxis="y1",
                                ),
                                row=1,
                                col=1,
                            )

                # Plot 2: Sentiment scores
                if "news_sentiment_mean" in symbol_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=symbol_df["Date"],
                            y=symbol_df["news_sentiment_mean"],
                            mode="lines",
                            name=f"{symbol} Sentiment",
                            line=dict(color=color, dash="dash"),
                            yaxis="y2",
                        ),
                        row=2,
                        col=1,
                    )

                # Plot 3: Risk predictions
                if predictions_df is not None and not predictions_df.empty:
                    pred_symbol_data = (
                        predictions_df[predictions_df["Symbol"] == symbol]
                        if "Symbol" in predictions_df.columns
                        else predictions_df
                    )
                    if (
                        not pred_symbol_data.empty
                        and "Risk_Probability" in pred_symbol_data.columns
                    ):
                        fig.add_trace(
                            go.Scatter(
                                x=pred_symbol_data["Date"],
                                y=pred_symbol_data["Risk_Probability"],
                                mode="lines",
                                name=f"{symbol} Risk Prob",
                                line=dict(color=color),
                                yaxis="y3",
                            ),
                            row=3,
                            col=1,
                        )

            # Update layout
            fig.update_layout(
                title="Market Risk Timeline Dashboard",
                showlegend=True,
                height=800,
                hovermode="x unified",
            )

            # Update y-axes
            fig.update_yaxes(title_text="Stock Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Sentiment Score", row=2, col=1)
            fig.update_yaxes(title_text="Risk Probability", row=3, col=1)
            fig.update_xaxes(title_text="Date", row=3, col=1)

            # Save plot
            output_path = os.path.join(
                self.output_dir, "risk_timeline_interactive.html"
            )
            fig.write_html(output_path)

            self.logger.info(f"Created interactive risk timeline: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Error creating interactive timeline: {str(e)}")
            return None

    def plot_risk_timeline_static(
        self,
        df: pd.DataFrame,
        symbols: List[str],
        predictions_df: Optional[pd.DataFrame] = None,
    ) -> Optional[str]:
        """Create static risk timeline using matplotlib"""

        if not self.matplotlib_available or df.empty:
            return None

        try:
            import matplotlib.dates as mdates
            import matplotlib.pyplot as plt

            # Select data for chosen symbols
            symbol_data = (
                df[df["Symbol"].isin(symbols)].copy()
                if "Symbol" in df.columns
                else df.copy()
            )

            if symbol_data.empty:
                return None

            if "Date" in symbol_data.columns:
                symbol_data["Date"] = pd.to_datetime(
                    symbol_data["Date"], errors="coerce"
                )
                symbol_data = symbol_data.dropna(subset=["Date"])
                symbol_data = symbol_data.sort_values("Date")

            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
            fig.suptitle(
                "Market Risk Timeline Dashboard", fontsize=16, fontweight="bold"
            )

            colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(symbols)))

            for i, symbol in enumerate(symbols[:10]):
                symbol_df = symbol_data[symbol_data["Symbol"] == symbol]
                if symbol_df.empty:
                    continue

                color = colors[i]

                # Plot 1: Stock prices and risk events
                if "Close" in symbol_df.columns and "Date" in symbol_df.columns:
                    axes[0].plot(
                        symbol_df["Date"],
                        symbol_df["Close"],
                        color=color,
                        label=f"{symbol} Price",
                        linewidth=1,
                    )

                    # Mark risk events
                    if "Risk_Label" in symbol_df.columns:
                        risk_events = symbol_df[symbol_df["Risk_Label"] == 1]
                        if not risk_events.empty:
                            axes[0].scatter(
                                risk_events["Date"],
                                risk_events["Close"],
                                color="red",
                                marker="^",
                                s=30,
                                alpha=0.7,
                                label=f"{symbol} Risk Events" if i == 0 else "",
                            )

                # Plot 2: Sentiment scores
                if "news_sentiment_mean" in symbol_df.columns:
                    sentiment_series = pd.to_numeric(
                        symbol_df["news_sentiment_mean"], errors="coerce"
                    )
                    if not sentiment_series.dropna().empty:
                        axes[1].plot(
                            symbol_df["Date"],
                            sentiment_series,
                            color=color,
                            alpha=0.7,
                            linewidth=1,
                        )

                # Plot 3: Risk predictions
                if predictions_df is not None and not predictions_df.empty:
                    pred_symbol_data = (
                        predictions_df[predictions_df["Symbol"] == symbol]
                        if "Symbol" in predictions_df.columns
                        else predictions_df
                    )
                    if (
                        not pred_symbol_data.empty
                        and "Risk_Probability" in pred_symbol_data.columns
                    ):
                        axes[2].plot(
                            pred_symbol_data["Date"],
                            pred_symbol_data["Risk_Probability"],
                            color=color,
                            linewidth=1,
                        )

            # Customize subplots
            axes[0].set_ylabel("Stock Price ($)")
            axes[0].set_title("Stock Prices and Risk Events")
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            axes[0].grid(True, alpha=0.3)

            axes[1].set_ylabel("Sentiment Score")
            axes[1].set_title("News Sentiment Scores")
            axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
            axes[1].grid(True, alpha=0.3)

            axes[2].set_ylabel("Risk Probability")
            axes[2].set_title("Model Risk Predictions")
            axes[2].axhline(
                y=0.5, color="red", linestyle="--", alpha=0.5, label="Risk Threshold"
            )
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            # Format x-axis
            axes[2].set_xlabel("Date")
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()

            # Save plot
            output_path = os.path.join(self.output_dir, "risk_timeline_static.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Created static risk timeline: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Error creating static timeline: {str(e)}")
            return None

    def plot_sentiment_analysis(
        self, df: pd.DataFrame, symbols: List[str]
    ) -> Optional[str]:
        """Create sentiment analysis visualizations"""

        if not self.matplotlib_available or df.empty:
            return None

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle("Sentiment Analysis Dashboard", fontsize=16, fontweight="bold")

            # Select data
            symbol_data = (
                df[df["Symbol"].isin(symbols)] if "Symbol" in df.columns else df
            )

            # Plot 1: Sentiment distribution
            if "news_sentiment_mean" in symbol_data.columns:
                axes[0, 0].hist(
                    symbol_data["news_sentiment_mean"].dropna(),
                    bins=50,
                    alpha=0.7,
                    color="blue",
                )
                axes[0, 0].axvline(x=0, color="red", linestyle="--", alpha=0.7)
                axes[0, 0].set_title("News Sentiment Distribution")
                axes[0, 0].set_xlabel("Sentiment Score")
                axes[0, 0].set_ylabel("Frequency")

            # Plot 2: Sentiment vs Returns correlation
            if all(
                col in symbol_data.columns for col in ["news_sentiment_mean", "Returns"]
            ):
                valid_data = symbol_data[["news_sentiment_mean", "Returns"]].dropna()
                if not valid_data.empty:
                    axes[0, 1].scatter(
                        valid_data["news_sentiment_mean"],
                        valid_data["Returns"],
                        alpha=0.5,
                        s=10,
                    )
                    axes[0, 1].set_title("Sentiment vs Stock Returns")
                    axes[0, 1].set_xlabel("Sentiment Score")
                    axes[0, 1].set_ylabel("Daily Returns")

                    # Add trend line
                    z = np.polyfit(
                        valid_data["news_sentiment_mean"], valid_data["Returns"], 1
                    )
                    p = np.poly1d(z)
                    axes[0, 1].plot(
                        valid_data["news_sentiment_mean"],
                        p(valid_data["news_sentiment_mean"]),
                        "r--",
                        alpha=0.8,
                    )

            # Plot 3: Sentiment by symbol (boxplot)
            if all(
                col in symbol_data.columns for col in ["Symbol", "news_sentiment_mean"]
            ):
                sentiment_by_symbol = []
                symbol_labels = []
                for symbol in symbols[:10]:
                    symbol_sentiment = symbol_data[symbol_data["Symbol"] == symbol][
                        "news_sentiment_mean"
                    ].dropna()
                    if not symbol_sentiment.empty:
                        sentiment_by_symbol.append(symbol_sentiment)
                        symbol_labels.append(symbol)

                if sentiment_by_symbol:
                    axes[1, 0].boxplot(sentiment_by_symbol, labels=symbol_labels)
                    axes[1, 0].set_title("Sentiment Distribution by Symbol")
                    axes[1, 0].set_xlabel("Symbol")
                    axes[1, 0].set_ylabel("Sentiment Score")
                    axes[1, 0].tick_params(axis="x", rotation=45)

            # Plot 4: Sentiment time series (average across all symbols)
            if all(
                col in symbol_data.columns for col in ["Date", "news_sentiment_mean"]
            ):
                daily_sentiment = (
                    symbol_data.groupby("Date")["news_sentiment_mean"]
                    .mean()
                    .reset_index()
                )

                axes[1, 1].plot(
                    daily_sentiment["Date"],
                    daily_sentiment["news_sentiment_mean"],
                    color="purple",
                    linewidth=1,
                )
                axes[1, 1].axhline(y=0, color="red", linestyle="--", alpha=0.5)
                axes[1, 1].set_title("Average Daily Sentiment")
                axes[1, 1].set_xlabel("Date")
                axes[1, 1].set_ylabel("Average Sentiment")
                axes[1, 1].tick_params(axis="x", rotation=45)

            plt.tight_layout()

            # Save plot
            output_path = os.path.join(self.output_dir, "sentiment_analysis.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Created sentiment analysis plot: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Error creating sentiment analysis plot: {str(e)}")
            return None

    def plot_feature_importance(
        self, df: pd.DataFrame, feature_importance: Optional[Dict[str, float]] = None
    ) -> Optional[str]:
        """Plot feature importance from models"""

        if not self.matplotlib_available:
            return None

        try:
            import matplotlib.pyplot as plt

            # If no feature importance provided, calculate correlation with risk
            if feature_importance is None and "Risk_Label" in df.columns:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                numeric_cols = [
                    col for col in numeric_cols if col not in ["Risk_Label", "Date"]
                ]

                correlations = {}
                for col in numeric_cols[:20]:  # Top 20 features
                    try:
                        corr = abs(df[col].corr(df["Risk_Label"]))
                        if not np.isnan(corr):
                            correlations[col] = corr
                    except:
                        continue

                feature_importance = dict(
                    sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:15]
                )

            if not feature_importance:
                return None

            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8))

            features = list(feature_importance.keys())
            importance_values = list(feature_importance.values())

            # Create horizontal bar plot
            y_pos = np.arange(len(features))
            bars = ax.barh(y_pos, importance_values, color="steelblue", alpha=0.7)

            # Customize plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.invert_yaxis()  # Top feature at the top
            ax.set_xlabel("Importance Score")
            ax.set_title(
                "Feature Importance for Risk Prediction", fontsize=14, fontweight="bold"
            )

            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(
                    width + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{importance_values[i]:.3f}",
                    ha="left",
                    va="center",
                )

            plt.tight_layout()

            # Save plot
            output_path = os.path.join(self.output_dir, "feature_importance.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Created feature importance plot: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Error creating feature importance plot: {str(e)}")
            return None

    def plot_risk_correlations(self, df: pd.DataFrame) -> Optional[str]:
        """Create correlation heatmap of risk-related features"""

        if not self.matplotlib_available or df.empty:
            return None

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Select risk-related features
            risk_features = []
            for col in df.columns:
                if any(
                    keyword in col.lower()
                    for keyword in [
                        "risk",
                        "volatility",
                        "sentiment",
                        "return",
                        "pe_ratio",
                        "debt",
                    ]
                ):
                    if df[col].dtype in ["int64", "float64"]:
                        risk_features.append(col)

            if len(risk_features) < 2:
                return None

            # Calculate correlation matrix
            corr_matrix = df[risk_features].corr()

            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 10))

            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap="RdYlBu_r",
                center=0,
                square=True,
                ax=ax,
                cbar_kws={"shrink": 0.8},
            )

            ax.set_title(
                "Risk Features Correlation Matrix", fontsize=14, fontweight="bold"
            )
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)

            plt.tight_layout()

            # Save plot
            output_path = os.path.join(self.output_dir, "risk_correlations.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Created risk correlations heatmap: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Error creating correlation heatmap: {str(e)}")
            return None

    def plot_model_performance(self, predictions_df: pd.DataFrame) -> Optional[str]:
        """Plot model performance metrics"""

        if not self.matplotlib_available or predictions_df.empty:
            return None

        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import auc, precision_recall_curve, roc_curve

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle("Model Performance Dashboard", fontsize=16, fontweight="bold")

            # Assuming predictions_df has actual labels and predictions
            if all(
                col in predictions_df.columns
                for col in ["Risk_Label", "Risk_Probability"]
            ):
                y_true = predictions_df["Risk_Label"]
                y_prob = predictions_df["Risk_Probability"]
                y_pred = (y_prob > 0.5).astype(int)

                # Plot 1: ROC Curve
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)

                axes[0, 0].plot(
                    fpr,
                    tpr,
                    color="darkorange",
                    lw=2,
                    label=f"ROC curve (AUC = {roc_auc:.2f})",
                )
                axes[0, 0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
                axes[0, 0].set_xlim([0.0, 1.0])
                axes[0, 0].set_ylim([0.0, 1.05])
                axes[0, 0].set_xlabel("False Positive Rate")
                axes[0, 0].set_ylabel("True Positive Rate")
                axes[0, 0].set_title("Receiver Operating Characteristic")
                axes[0, 0].legend(loc="lower right")

                # Plot 2: Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                pr_auc = auc(recall, precision)

                axes[0, 1].plot(
                    recall,
                    precision,
                    color="blue",
                    lw=2,
                    label=f"PR curve (AUC = {pr_auc:.2f})",
                )
                axes[0, 1].set_xlabel("Recall")
                axes[0, 1].set_ylabel("Precision")
                axes[0, 1].set_title("Precision-Recall Curve")
                axes[0, 1].legend()

                # Plot 3: Prediction distribution
                axes[1, 0].hist(
                    y_prob[y_true == 0],
                    bins=50,
                    alpha=0.7,
                    label="Non-Risk",
                    color="blue",
                )
                axes[1, 0].hist(
                    y_prob[y_true == 1], bins=50, alpha=0.7, label="Risk", color="red"
                )
                axes[1, 0].axvline(
                    x=0.5, color="black", linestyle="--", label="Threshold"
                )
                axes[1, 0].set_xlabel("Risk Probability")
                axes[1, 0].set_ylabel("Frequency")
                axes[1, 0].set_title("Risk Probability Distribution")
                axes[1, 0].legend()

                # Plot 4: Confusion Matrix
                from sklearn.metrics import confusion_matrix

                cm = confusion_matrix(y_true, y_pred)

                im = axes[1, 1].imshow(cm, interpolation="nearest", cmap="Blues")
                axes[1, 1].set_title("Confusion Matrix")

                # Add text annotations
                thresh = cm.max() / 2.0
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        axes[1, 1].text(
                            j,
                            i,
                            format(cm[i, j], "d"),
                            ha="center",
                            va="center",
                            color="white" if cm[i, j] > thresh else "black",
                        )

                axes[1, 1].set_ylabel("True Label")
                axes[1, 1].set_xlabel("Predicted Label")
                axes[1, 1].set_xticks([0, 1])
                axes[1, 1].set_yticks([0, 1])
                axes[1, 1].set_xticklabels(["Non-Risk", "Risk"])
                axes[1, 1].set_yticklabels(["Non-Risk", "Risk"])

            plt.tight_layout()

            # Save plot
            output_path = os.path.join(self.output_dir, "model_performance.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Created model performance plot: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Error creating model performance plot: {str(e)}")
            return None

    def plot_market_regimes(self, df: pd.DataFrame) -> Optional[str]:
        """Create market regime analysis visualization"""

        if not self.plotly_available or df.empty:
            return None

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Create market-wide metrics
            if "Date" not in df.columns:
                return None

            market_data = (
                df.groupby("Date")
                .agg(
                    {
                        "Market_Return": "first",
                        "Market_Volatility": "first",
                        "news_sentiment_mean": "mean",
                        "Risk_Label": "mean",
                    }
                )
                .reset_index()
            )

            # Create subplots
            fig = make_subplots(
                rows=4,
                cols=1,
                shared_xaxes=True,
                subplot_titles=[
                    "Market Returns",
                    "Market Volatility",
                    "Average Sentiment",
                    "Risk Level",
                ],
                vertical_spacing=0.05,
            )

            # Plot market returns
            if "Market_Return" in market_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=market_data["Date"],
                        y=market_data["Market_Return"],
                        mode="lines",
                        name="Market Returns",
                        line=dict(color="blue"),
                    ),
                    row=1,
                    col=1,
                )

                # Add recession periods (negative returns)
                negative_returns = market_data[market_data["Market_Return"] < -0.02]
                if not negative_returns.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=negative_returns["Date"],
                            y=negative_returns["Market_Return"],
                            mode="markers",
                            name="High Negative Returns",
                            marker=dict(color="red", size=6),
                        ),
                        row=1,
                        col=1,
                    )

            # Plot market volatility
            if "Market_Volatility" in market_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=market_data["Date"],
                        y=market_data["Market_Volatility"],
                        mode="lines",
                        name="Market Volatility",
                        line=dict(color="orange"),
                    ),
                    row=2,
                    col=1,
                )

            # Plot average sentiment
            if "news_sentiment_mean" in market_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=market_data["Date"],
                        y=market_data["news_sentiment_mean"],
                        mode="lines",
                        name="Average Sentiment",
                        line=dict(color="green"),
                    ),
                    row=3,
                    col=1,
                )

                # Add zero line
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

            # Plot risk level
            if "Risk_Label" in market_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=market_data["Date"],
                        y=market_data["Risk_Label"],
                        mode="lines+markers",
                        name="Risk Level",
                        line=dict(color="red"),
                    ),
                    row=4,
                    col=1,
                )

                # Add risk threshold
                fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=4, col=1)

            # Update layout
            fig.update_layout(
                title="Market Regime Analysis",
                showlegend=True,
                height=800,
                hovermode="x unified",
            )

            # Update y-axes
            fig.update_yaxes(title_text="Returns", row=1, col=1)
            fig.update_yaxes(title_text="Volatility", row=2, col=1)
            fig.update_yaxes(title_text="Sentiment", row=3, col=1)
            fig.update_yaxes(title_text="Risk Level", row=4, col=1)
            fig.update_xaxes(title_text="Date", row=4, col=1)

            # Save plot
            output_path = os.path.join(self.output_dir, "market_regimes.html")
            fig.write_html(output_path)

            self.logger.info(f"Created market regimes plot: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Error creating market regimes plot: {str(e)}")
            return None

    def create_summary_report(self, visualization_paths: Dict[str, str]) -> str:
        """Create HTML summary report with all visualizations"""

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Market Risk Analysis Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; text-align: center; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                .visualization {{ margin: 30px 0; text-align: center; }}
                .visualization img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                .description {{ margin: 10px 0; color: #555; font-style: italic; }}
                .timestamp {{ text-align: center; color: #999; margin-top: 40px; }}
            </style>
        </head>
        <body>
            <h1>Market Risk Analysis Dashboard</h1>
            <p style="text-align: center; color: #666;">
                Comprehensive analysis of market risk using stock prices, news sentiment, and SEC filings
            </p>
        """

        # Add each visualization
        viz_descriptions = {
            "risk_timeline_interactive": "Interactive timeline showing stock prices, risk events, sentiment scores, and model predictions",
            "risk_timeline_static": "Static timeline view of risk events and stock price movements",
            "sentiment_analysis": "Analysis of news sentiment patterns and correlation with stock returns",
            "feature_importance": "Most important features for predicting market risk",
            "risk_correlations": "Correlation matrix of risk-related features",
            "model_performance": "Machine learning model performance metrics and validation",
            "market_regimes": "Interactive analysis of different market regimes and risk periods",
            "explainability_shap_global": "Global SHAP feature importance highlighting drivers of risk",
            "explainability_shap_local": "Local SHAP explanation for an exemplar scenario",
            "explainability_lime": "LIME-based local explanation with human-readable feature weights",
            "cews_timeline": "Crisis Early Warning Score timeline balancing early detection and false alarms",
        }

        for viz_name, viz_path in visualization_paths.items():
            if os.path.exists(viz_path):
                description = viz_descriptions.get(viz_name, "Analysis visualization")

                html_content += f"""
                <div class="visualization">
                    <h2>{viz_name.replace('_', ' ').title()}</h2>
                    <p class="description">{description}</p>
                """

                # Embed different file types
                if viz_path.endswith(".html"):
                    # For interactive plots, create a link
                    html_content += f'<p><a href="{os.path.basename(viz_path)}" target="_blank">View Interactive Plot</a></p>'
                elif viz_path.endswith(".png"):
                    # For static images, embed directly
                    html_content += (
                        f'<img src="{os.path.basename(viz_path)}" alt="{viz_name}">'
                    )

                html_content += "</div>"

        html_content += f"""
            <div class="timestamp">
                <p>Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """

        # Save HTML report
        report_path = os.path.join(self.output_dir, "risk_analysis_report.html")
        with open(report_path, "w") as f:
            f.write(html_content)

        self.logger.info(f"Created summary report: {report_path}")
        return report_path

    def create_cews_dashboard(
        self,
        predictions_df: pd.DataFrame,
        probability_col: str = "Risk_Probability",
        label_col: str = "Risk_Label",
        date_col: str = "Date",
        symbol_col: str = "Symbol",
        threshold: float = 0.5,
        lookback_window: int = 5,
    ) -> Dict[str, str]:
        """Generate CEWS timeline and summary artifacts."""

        outputs: Dict[str, str] = {}

        if predictions_df.empty:
            self.logger.warning("Prediction DataFrame empty; skipping CEWS dashboard")
            return outputs

        required_cols = {probability_col, label_col, date_col}
        if not required_cols.issubset(predictions_df.columns):
            self.logger.warning(
                "Prediction DataFrame missing required columns for CEWS: %s",
                required_cols - set(predictions_df.columns),
            )
            return outputs

        try:
            cews_result = compute_cews_score(
                predictions_df,
                probability_col=probability_col,
                label_col=label_col,
                date_col=date_col,
                symbol_col=symbol_col if symbol_col in predictions_df.columns else None,
                threshold=threshold,
                lookback_window=lookback_window,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            self.logger.error("Failed to compute CEWS: %s", exc, exc_info=True)
            return outputs

        if self.plotly_available and not cews_result.timeline.empty:
            timeline_path = self._plot_cews_timeline(cews_result)
            if timeline_path:
                outputs["cews_timeline"] = timeline_path

        summary_path = self._write_cews_summary(cews_result)
        if summary_path:
            outputs["cews_summary"] = summary_path

        return outputs

    def _plot_cews_timeline(self, cews_result: CEWSResult) -> Optional[str]:
        """Create interactive CEWS timeline plot."""

        if not self.plotly_available or cews_result.timeline.empty:
            return None

        try:
            import plotly.graph_objects as go

            timeline = cews_result.timeline.copy()
            date_series = timeline.iloc[:, 0]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=date_series,
                    y=timeline["cews"],
                    mode="lines",
                    name="CEWS",
                    line=dict(color="#2563eb", width=3),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=date_series,
                    y=timeline["early_reward"],
                    mode="lines",
                    name="Early Detection Reward",
                    line=dict(color="#16a34a", width=2, dash="dash"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=date_series,
                    y=timeline["false_alarm_penalty"],
                    mode="lines",
                    name="False Alarm Penalty",
                    line=dict(color="#dc2626", width=2, dash="dot"),
                )
            )

            fig.update_layout(
                title="Crisis Early Warning Score Timeline",
                xaxis_title="Date",
                yaxis_title="Score",
                hovermode="x unified",
                template="plotly_white",
                legend=dict(orientation="h", y=1.08, x=0),
                yaxis=dict(range=[0, 1]),
            )

            output_path = os.path.join(self.output_dir, "cews_timeline.html")
            fig.write_html(output_path)
            self.logger.info("Created CEWS timeline: %s", output_path)
            return output_path

        except Exception as exc:  # pragma: no cover - defensive guard
            self.logger.error("Error creating CEWS timeline: %s", exc, exc_info=True)
            return None

    def _write_cews_summary(self, cews_result: CEWSResult) -> Optional[str]:
        """Persist CEWS summary metrics for downstream dashboards."""

        try:
            summary = {
                "score": cews_result.score,
                "early_detection_reward": cews_result.early_detection_reward,
                "false_alarm_penalty": cews_result.false_alarm_penalty,
                "metadata": cews_result.metadata,
            }
            summary_path = os.path.join(self.output_dir, "cews_summary.json")
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
            self.logger.info("Saved CEWS summary: %s", summary_path)
            return summary_path
        except Exception as exc:  # pragma: no cover - defensive guard
            self.logger.error("Failed to save CEWS summary: %s", exc, exc_info=True)
            return None

    def create_explainability_dashboard(
        self,
        model: Any,
        feature_df: pd.DataFrame,
        sample_index: int = 0,
        class_names: Optional[np.ndarray] = None,
    ) -> Dict[str, str]:
        """Generate explainability visualizations using SHAP and LIME."""

        outputs: Dict[str, str] = {}

        if model is None:
            self.logger.warning("No model supplied for explainability dashboard")
            return outputs

        if feature_df is None or feature_df.empty:
            self.logger.warning(
                "Empty feature frame supplied for explainability dashboard"
            )
            return outputs

        plt_module = None
        if self.matplotlib_available:
            try:
                import matplotlib.pyplot as plt  # type: ignore

                plt_module = plt
            except ImportError:
                self.logger.warning(
                    "Matplotlib unavailable at runtime; falling back to tabular explainability outputs"
                )

        def _write_table_html(df: pd.DataFrame, title: str, filename: str) -> str:
            table_path = os.path.join(self.output_dir, filename)
            html = (
                "<html><head><meta charset='utf-8'><title>{title}</title></head>"
                "<body><h2>{title}</h2>{table}</body></html>"
            ).format(title=title, table=df.to_html(index=False))
            with open(table_path, "w", encoding="utf-8") as handle:
                handle.write(html)
            return table_path

        try:
            global_importance = compute_global_shap_importance(model, feature_df)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error(
                f"Error computing global SHAP importance: {exc}", exc_info=True
            )
            global_importance = None

        if global_importance is not None and not global_importance.empty:
            top_global = global_importance.head(20)
            if plt_module is not None:
                fig, ax = plt_module.subplots(figsize=(12, 8))
                bars = ax.barh(
                    top_global["feature"][::-1],
                    top_global["importance"][::-1],
                    color="#1f77b4",
                    alpha=0.8,
                )
                ax.set_title(
                    "Global Feature Importance (Mean |SHAP|)",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.set_xlabel("Contribution to risk prediction")
                ax.set_ylabel("Feature")
                ax.invert_yaxis()
                for bar, value in zip(bars, top_global["importance"][::-1]):
                    ax.text(
                        value + 0.001,
                        bar.get_y() + bar.get_height() / 2,
                        f"{value:.4f}",
                        va="center",
                        fontsize=9,
                    )
                fig.tight_layout()
                global_path = os.path.join(
                    self.output_dir, "explainability_shap_global.png"
                )
                fig.savefig(global_path, dpi=300, bbox_inches="tight")
                plt_module.close(fig)
            else:
                global_path = _write_table_html(
                    top_global,
                    "Global Feature Importance (Mean |SHAP|)",
                    "explainability_shap_global.html",
                )
            outputs["explainability_shap_global"] = global_path

        try:
            local_explanation = compute_local_shap_explanation(
                model, feature_df, sample_index
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error(
                f"Error computing local SHAP explanation: {exc}", exc_info=True
            )
            local_explanation = None

        if local_explanation is not None and not local_explanation.empty:
            top_local = local_explanation.head(20).copy()
            plot_frame = top_local.iloc[::-1]
            if plt_module is not None:
                fig, ax = plt_module.subplots(figsize=(12, 8))
                colors = [
                    "#2ecc71" if val >= 0 else "#e74c3c"
                    for val in plot_frame["shap_value"]
                ]
                ax.barh(
                    plot_frame["feature"],
                    plot_frame["shap_value"],
                    color=colors,
                    alpha=0.85,
                )
                ax.axvline(0, color="#333", linewidth=1)
                ax.set_title("Local SHAP Contributions", fontsize=14, fontweight="bold")
                ax.set_xlabel("Impact on predicted risk")
                ax.set_ylabel("Feature")
                fig.tight_layout()
                local_path = os.path.join(
                    self.output_dir, "explainability_shap_local.png"
                )
                fig.savefig(local_path, dpi=300, bbox_inches="tight")
                plt_module.close(fig)
            else:
                local_path = _write_table_html(
                    top_local,
                    "Local SHAP Contributions",
                    "explainability_shap_local.html",
                )
            outputs["explainability_shap_local"] = local_path

        try:
            lime_explanation = compute_lime_explanation(
                model, feature_df, sample_index, class_names=class_names
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error(f"Error computing LIME explanation: {exc}", exc_info=True)
            lime_explanation = None

        if lime_explanation is not None and not lime_explanation.empty:
            top_lime = lime_explanation.head(20)
            if plt_module is not None:
                fig, ax = plt_module.subplots(figsize=(12, 6))
                sorted_lime = top_lime.sort_values("weight")
                colors = [
                    "#2ecc71" if val >= 0 else "#e74c3c"
                    for val in sorted_lime["weight"]
                ]
                ax.barh(
                    sorted_lime["feature"],
                    sorted_lime["weight"],
                    color=colors,
                    alpha=0.85,
                )
                ax.axvline(0, color="#333", linewidth=1)
                ax.set_title("LIME Local Explanation", fontsize=14, fontweight="bold")
                ax.set_xlabel("Weight toward predicted class")
                ax.set_ylabel("Feature")
                fig.tight_layout()
                lime_path = os.path.join(self.output_dir, "explainability_lime.png")
                fig.savefig(lime_path, dpi=300, bbox_inches="tight")
                plt_module.close(fig)
            else:
                lime_path = _write_table_html(
                    top_lime,
                    "LIME Local Explanation",
                    "explainability_lime.html",
                )
            outputs["explainability_lime"] = lime_path

        if not outputs:
            self.logger.warning(
                "Explainability artifacts could not be generated; check optional dependencies"
            )
        else:
            self.logger.info("Created %s explainability artifacts", len(outputs))

        return outputs
