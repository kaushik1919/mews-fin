"""
Enhanced Risk Timeline Visualizer
Optimized interactive risk timeline with better performance and insights
"""

import logging
import os
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class EnhancedRiskTimelineVisualizer:
    """Creates optimized interactive risk timeline visualizations"""

    def __init__(self, output_dir: str = None):
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir or "outputs"
        os.makedirs(self.output_dir, exist_ok=True)

        # Check for required libraries
        self.plotly_available = self._check_plotly()

    def _check_plotly(self) -> bool:
        """Check if plotly is available"""
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            import plotly.offline as pyo
            from plotly.subplots import make_subplots

            return True
        except ImportError:
            self.logger.warning("Plotly not available - interactive plots disabled")
            return False

    def create_enhanced_risk_timeline(
        self,
        df: pd.DataFrame,
        symbols: List[str] = None,
        predictions_df: pd.DataFrame = None,
        show_confidence_bands: bool = True,
        show_regime_changes: bool = True,
    ) -> Optional[str]:
        """
        Create enhanced interactive risk timeline with optimized performance

        Args:
            df: DataFrame with enhanced risk features
            symbols: List of symbols to visualize
            predictions_df: DataFrame with model predictions
            show_confidence_bands: Whether to show prediction confidence intervals
            show_regime_changes: Whether to highlight market regime changes

        Returns:
            Path to generated HTML file
        """
        if not self.plotly_available or df.empty:
            return None

        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            self.logger.info("Creating enhanced risk timeline...")

            # Prepare data
            if symbols is None:
                symbols = df["Symbol"].unique()[:8]  # Limit for performance

            # Filter data for selected symbols
            filtered_df = (
                df[df["Symbol"].isin(symbols)] if "Symbol" in df.columns else df
            )

            if filtered_df.empty:
                return None

            # Sort by date for proper time series
            if "Date" in filtered_df.columns:
                filtered_df = filtered_df.sort_values("Date")

            # Create optimized subplot structure
            fig = make_subplots(
                rows=4,
                cols=1,
                shared_xaxes=True,
                subplot_titles=[
                    "📈 Stock Prices & Risk Events",
                    "💭 Sentiment Analysis (News + SEC)",
                    "📊 Risk Probability Predictions",
                    "🎯 Composite Risk Score",
                ],
                vertical_spacing=0.04,
                row_heights=[0.35, 0.25, 0.25, 0.15],
                specs=[
                    [{"secondary_y": True}],
                    [{"secondary_y": True}],
                    [{"secondary_y": False}],
                    [{"secondary_y": False}],
                ],
            )

            # Color palette for better visualization
            colors = px.colors.qualitative.Set3[: len(symbols)]

            for i, symbol in enumerate(symbols):
                symbol_df = filtered_df[filtered_df["Symbol"] == symbol].copy()
                if symbol_df.empty:
                    continue

                color = colors[i % len(colors)]

                # 1. STOCK PRICES & RISK EVENTS
                if all(col in symbol_df.columns for col in ["Date", "Close"]):
                    # Stock price line
                    fig.add_trace(
                        go.Scatter(
                            x=symbol_df["Date"],
                            y=symbol_df["Close"],
                            mode="lines",
                            name=f"{symbol} Price",
                            line=dict(color=color, width=2),
                            hovertemplate=f"<b>{symbol}</b><br>"
                            + "Date: %{x}<br>"
                            + "Price: $%{y:.2f}<br>"
                            + "<extra></extra>",
                            showlegend=True,
                        ),
                        row=1,
                        col=1,
                    )

                    # Risk events as markers
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
                                        color="red",
                                        size=10,
                                        symbol="triangle-up",
                                        line=dict(color="darkred", width=1),
                                    ),
                                    hovertemplate=f"<b>{symbol} Risk Event</b><br>"
                                    + "Date: %{x}<br>"
                                    + "Price: $%{y:.2f}<br>"
                                    + "<extra></extra>",
                                    showlegend=False,
                                ),
                                row=1,
                                col=1,
                            )

                    # Support and resistance levels
                    if (
                        "support_level" in symbol_df.columns
                        and "resistance_level" in symbol_df.columns
                    ):
                        fig.add_trace(
                            go.Scatter(
                                x=symbol_df["Date"],
                                y=symbol_df["support_level"],
                                mode="lines",
                                name=f"{symbol} Support",
                                line=dict(color=color, dash="dash", width=1),
                                opacity=0.5,
                                showlegend=False,
                            ),
                            row=1,
                            col=1,
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=symbol_df["Date"],
                                y=symbol_df["resistance_level"],
                                mode="lines",
                                name=f"{symbol} Resistance",
                                line=dict(color=color, dash="dot", width=1),
                                opacity=0.5,
                                showlegend=False,
                            ),
                            row=1,
                            col=1,
                        )

                # 2. SENTIMENT ANALYSIS
                # News sentiment
                if "news_sentiment_mean" in symbol_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=symbol_df["Date"],
                            y=symbol_df["news_sentiment_mean"],
                            mode="lines",
                            name=f"{symbol} News Sentiment",
                            line=dict(color=color, width=2),
                            hovertemplate=f"<b>{symbol} News Sentiment</b><br>"
                            + "Date: %{x}<br>"
                            + "Sentiment: %{y:.3f}<br>"
                            + "<extra></extra>",
                            showlegend=True,
                        ),
                        row=2,
                        col=1,
                    )

                # SEC sentiment
                if "sec_sentiment_mean" in symbol_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=symbol_df["Date"],
                            y=symbol_df["sec_sentiment_mean"],
                            mode="lines",
                            name=f"{symbol} SEC Sentiment",
                            line=dict(color=color, dash="dash", width=2),
                            hovertemplate=f"<b>{symbol} SEC Sentiment</b><br>"
                            + "Date: %{x}<br>"
                            + "Sentiment: %{y:.3f}<br>"
                            + "<extra></extra>",
                            showlegend=True,
                        ),
                        row=2,
                        col=1,
                        secondary_y=True,
                    )

                # 3. RISK PREDICTIONS
                if predictions_df is not None and not predictions_df.empty:
                    pred_data = (
                        predictions_df[predictions_df["Symbol"] == symbol]
                        if "Symbol" in predictions_df.columns
                        else predictions_df
                    )

                    if not pred_data.empty and "Risk_Probability" in pred_data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=pred_data["Date"],
                                y=pred_data["Risk_Probability"],
                                mode="lines",
                                name=f"{symbol} Risk Probability",
                                line=dict(color=color, width=3),
                                fill="tonexty" if i == 0 else None,
                                fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.1])}",
                                hovertemplate=f"<b>{symbol} Risk Probability</b><br>"
                                + "Date: %{x}<br>"
                                + "Risk: %{y:.1%}<br>"
                                + "<extra></extra>",
                                showlegend=True,
                            ),
                            row=3,
                            col=1,
                        )

                        # Add confidence bands if available
                        if (
                            show_confidence_bands
                            and "Risk_Probability_Lower" in pred_data.columns
                        ):
                            fig.add_trace(
                                go.Scatter(
                                    x=pred_data["Date"],
                                    y=pred_data["Risk_Probability_Upper"],
                                    mode="lines",
                                    line=dict(width=0),
                                    showlegend=False,
                                    hoverinfo="skip",
                                ),
                                row=3,
                                col=1,
                            )

                            fig.add_trace(
                                go.Scatter(
                                    x=pred_data["Date"],
                                    y=pred_data["Risk_Probability_Lower"],
                                    mode="lines",
                                    line=dict(width=0),
                                    fill="tonexty",
                                    fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.1])}",
                                    showlegend=False,
                                    hoverinfo="skip",
                                ),
                                row=3,
                                col=1,
                            )

                # 4. COMPOSITE RISK SCORE
                if "composite_risk_score" in symbol_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=symbol_df["Date"],
                            y=symbol_df["composite_risk_score"],
                            mode="lines",
                            name=f"{symbol} Composite Risk",
                            line=dict(color=color, width=2),
                            hovertemplate=f"<b>{symbol} Composite Risk</b><br>"
                            + "Date: %{x}<br>"
                            + "Risk Score: %{y:.3f}<br>"
                            + "<extra></extra>",
                            showlegend=True,
                        ),
                        row=4,
                        col=1,
                    )

            # Add market regime backgrounds if available
            if show_regime_changes and "bull_market" in filtered_df.columns:
                self._add_regime_backgrounds(fig, filtered_df)

            # Add risk threshold lines
            self._add_risk_thresholds(fig)

            # Update layout for better performance and appearance
            fig.update_layout(
                title={
                    "text": "🔍 Enhanced Market Risk Timeline Dashboard",
                    "x": 0.5,
                    "xanchor": "center",
                    "font": {"size": 24},
                },
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                height=1000,
                hovermode="x unified",
                template="plotly_white",
                # Performance optimizations
                dragmode="pan",
                uirevision=True,
            )

            # Update axes
            fig.update_yaxes(title_text="Stock Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="News Sentiment", row=2, col=1)
            fig.update_yaxes(title_text="SEC Sentiment", row=2, col=1, secondary_y=True)
            fig.update_yaxes(
                title_text="Risk Probability", row=3, col=1, tickformat=".0%"
            )
            fig.update_yaxes(title_text="Composite Risk", row=4, col=1, range=[0, 1])
            fig.update_xaxes(title_text="Date", row=4, col=1)

            # Add range selector for time navigation
            fig.update_layout(
                xaxis4=dict(
                    rangeselector=dict(
                        buttons=list(
                            [
                                dict(
                                    count=1,
                                    label="1M",
                                    step="month",
                                    stepmode="backward",
                                ),
                                dict(
                                    count=3,
                                    label="3M",
                                    step="month",
                                    stepmode="backward",
                                ),
                                dict(
                                    count=6,
                                    label="6M",
                                    step="month",
                                    stepmode="backward",
                                ),
                                dict(
                                    count=1,
                                    label="1Y",
                                    step="year",
                                    stepmode="backward",
                                ),
                                dict(step="all"),
                            ]
                        )
                    ),
                    rangeslider=dict(visible=False),
                    type="date",
                )
            )

            # Save with optimization
            output_path = os.path.join(
                self.output_dir, "enhanced_risk_timeline_interactive.html"
            )

            # Configure for better performance
            config = {
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "risk_timeline",
                    "height": 1000,
                    "width": 1400,
                    "scale": 2,
                },
            }

            fig.write_html(
                output_path,
                config=config,
                include_plotlyjs="cdn",  # Use CDN for smaller file size
                div_id="risk-timeline-div",
            )

            self.logger.info(f"Enhanced risk timeline created: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Error creating enhanced timeline: {str(e)}")
            return None

    def _add_regime_backgrounds(self, fig, df: pd.DataFrame):
        """Add colored backgrounds for different market regimes"""

        if "Date" not in df.columns:
            return

        # Get unique dates for regime changes
        dates = df["Date"].unique()

        # Add bull market periods as green background
        if "bull_market" in df.columns:
            bull_periods = df[df["bull_market"] == 1]["Date"].unique()
            if len(bull_periods) > 0:
                for i in range(len(dates) - 1):
                    if dates[i] in bull_periods:
                        fig.add_vrect(
                            x0=dates[i],
                            x1=dates[i + 1],
                            fillcolor="green",
                            opacity=0.1,
                            layer="below",
                            line_width=0,
                            row=1,
                            col=1,
                        )

        # Add bear market periods as red background
        if "bear_market" in df.columns:
            bear_periods = df[df["bear_market"] == 1]["Date"].unique()
            if len(bear_periods) > 0:
                for i in range(len(dates) - 1):
                    if dates[i] in bear_periods:
                        fig.add_vrect(
                            x0=dates[i],
                            x1=dates[i + 1],
                            fillcolor="red",
                            opacity=0.1,
                            layer="below",
                            line_width=0,
                            row=1,
                            col=1,
                        )

    def _add_risk_thresholds(self, fig):
        """Add horizontal lines for risk thresholds"""

        # Risk probability thresholds
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="orange",
            annotation_text="Medium Risk (50%)",
            row=3,
            col=1,
        )

        fig.add_hline(
            y=0.7,
            line_dash="dash",
            line_color="red",
            annotation_text="High Risk (70%)",
            row=3,
            col=1,
        )

        # Composite risk score thresholds
        fig.add_hline(
            y=0.6,
            line_dash="dash",
            line_color="orange",
            annotation_text="Caution",
            row=4,
            col=1,
        )

        fig.add_hline(
            y=0.8,
            line_dash="dash",
            line_color="red",
            annotation_text="Alert",
            row=4,
            col=1,
        )

    def create_risk_summary_dashboard(
        self, df: pd.DataFrame, symbols: List[str] = None
    ) -> Optional[str]:
        """
        Create a summary dashboard with key risk metrics

        Args:
            df: DataFrame with enhanced risk features
            symbols: List of symbols to analyze

        Returns:
            Path to generated HTML file
        """
        if not self.plotly_available or df.empty:
            return None

        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            self.logger.info("Creating risk summary dashboard...")

            if symbols is None:
                symbols = df["Symbol"].unique()[:10]

            # Create summary metrics
            summary_data = []

            for symbol in symbols:
                symbol_df = df[df["Symbol"] == symbol]
                if symbol_df.empty:
                    continue

                # Calculate summary metrics
                latest_data = symbol_df.iloc[-1] if not symbol_df.empty else None

                if latest_data is not None:
                    summary_data.append(
                        {
                            "Symbol": symbol,
                            "Current_Risk": latest_data.get("composite_risk_score", 0),
                            "Sentiment_Score": latest_data.get(
                                "news_sentiment_mean", 0
                            ),
                            "Volatility_Risk": latest_data.get(
                                "volatility_risk_score", 0
                            ),
                            "Technical_Risk": latest_data.get(
                                "technical_risk_score", 0
                            ),
                            "Price": latest_data.get("Close", 0),
                        }
                    )

            if not summary_data:
                return None

            summary_df = pd.DataFrame(summary_data)

            # Create dashboard
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=[
                    "Current Risk Levels by Symbol",
                    "Risk vs Sentiment Scatter",
                    "Risk Component Breakdown",
                    "Top Risk Stocks",
                ],
                specs=[
                    [{"type": "bar"}, {"type": "scatter"}],
                    [{"type": "radar"}, {"type": "table"}],
                ],
            )

            # Risk levels bar chart
            fig.add_trace(
                go.Bar(
                    x=summary_df["Symbol"],
                    y=summary_df["Current_Risk"],
                    marker_color=summary_df["Current_Risk"],
                    colorscale="RdYlGn_r",
                    name="Current Risk",
                ),
                row=1,
                col=1,
            )

            # Risk vs Sentiment scatter
            fig.add_trace(
                go.Scatter(
                    x=summary_df["Sentiment_Score"],
                    y=summary_df["Current_Risk"],
                    mode="markers+text",
                    text=summary_df["Symbol"],
                    textposition="top center",
                    marker=dict(
                        size=10,
                        color=summary_df["Current_Risk"],
                        colorscale="RdYlGn_r",
                        showscale=True,
                    ),
                    name="Risk vs Sentiment",
                ),
                row=1,
                col=2,
            )

            # Top risk stocks table
            top_risk = summary_df.nlargest(5, "Current_Risk")

            fig.add_trace(
                go.Table(
                    header=dict(
                        values=["Symbol", "Risk Score", "Price", "Sentiment"],
                        fill_color="lightblue",
                        align="center",
                    ),
                    cells=dict(
                        values=[
                            top_risk["Symbol"],
                            [f"{x:.3f}" for x in top_risk["Current_Risk"]],
                            [f"${x:.2f}" for x in top_risk["Price"]],
                            [f"{x:.3f}" for x in top_risk["Sentiment_Score"]],
                        ],
                        fill_color="white",
                        align="center",
                    ),
                ),
                row=2,
                col=2,
            )

            fig.update_layout(
                title="📊 Risk Summary Dashboard", height=800, showlegend=False
            )

            # Save dashboard
            output_path = os.path.join(self.output_dir, "risk_summary_dashboard.html")
            fig.write_html(output_path, include_plotlyjs="cdn")

            self.logger.info(f"Risk summary dashboard created: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Error creating summary dashboard: {str(e)}")
            return None
