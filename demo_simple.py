"""
Simple demo showcasing the MEWS system with existing data
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_demo_visualization():
    """Create a demo visualization showing the MEWS capabilities"""

    print("🚀 MEWS System Demo - Risk Timeline Visualization")
    print("=" * 60)

    try:
        # Load existing data
        data_path = "data/integrated_dataset.csv"

        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df["Date"] = pd.to_datetime(df["Date"])

            print(f"📊 Dataset loaded: {len(df)} rows, {len(df.columns)} columns")

            # Focus on a few symbols for demo
            symbols = ["AAPL", "MSFT", "GOOGL"]
            demo_df = df[df["Symbol"].isin(symbols)].copy()

            print(f"🏢 Demo symbols: {symbols}")
            print(f"📅 Date range: {demo_df['Date'].min()} to {demo_df['Date'].max()}")

            # Create enhanced demo timeline
            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                subplot_titles=[
                    "📈 Stock Prices & Technical Indicators",
                    "⚡ Volatility & Volume Analysis",
                    "🎯 Risk Indicators & Signals",
                ],
                vertical_spacing=0.08,
                row_heights=[0.4, 0.3, 0.3],
            )

            colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green

            for i, symbol in enumerate(symbols):
                symbol_data = demo_df[demo_df["Symbol"] == symbol].sort_values("Date")

                if symbol_data.empty:
                    continue

                color = colors[i]

                # Plot 1: Stock prices with moving averages
                fig.add_trace(
                    go.Scatter(
                        x=symbol_data["Date"],
                        y=symbol_data["Close"],
                        mode="lines",
                        name=f"{symbol} Price",
                        line=dict(color=color, width=2),
                        hovertemplate=f"<b>{symbol}</b><br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

                # Add SMA if available
                if "SMA_20" in symbol_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=symbol_data["Date"],
                            y=symbol_data["SMA_20"],
                            mode="lines",
                            name=f"{symbol} SMA20",
                            line=dict(color=color, dash="dash", width=1),
                            opacity=0.7,
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                    )

                # Plot 2: Volatility analysis
                if "volatility" in symbol_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=symbol_data["Date"],
                            y=symbol_data["volatility"],
                            mode="lines",
                            name=f"{symbol} Volatility",
                            line=dict(color=color, width=2),
                            hovertemplate=f"<b>{symbol} Volatility</b><br>Date: %{{x}}<br>Vol: %{{y:.4f}}<extra></extra>",
                        ),
                        row=2,
                        col=1,
                    )

                # Plot 3: RSI as risk indicator
                if "RSI" in symbol_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=symbol_data["Date"],
                            y=symbol_data["RSI"],
                            mode="lines",
                            name=f"{symbol} RSI",
                            line=dict(color=color, width=2),
                            hovertemplate=f"<b>{symbol} RSI</b><br>Date: %{{x}}<br>RSI: %{{y:.1f}}<extra></extra>",
                        ),
                        row=3,
                        col=1,
                    )

            # Add RSI overbought/oversold lines
            fig.add_hline(
                y=70,
                line_dash="dash",
                line_color="red",
                annotation_text="Overbought (70)",
                row=3,
                col=1,
            )
            fig.add_hline(
                y=30,
                line_dash="dash",
                line_color="green",
                annotation_text="Oversold (30)",
                row=3,
                col=1,
            )

            # Update layout
            fig.update_layout(
                title={
                    "text": "🔍 MEWS Risk Analysis Dashboard - Live Demo",
                    "x": 0.5,
                    "xanchor": "center",
                    "font": {"size": 20},
                },
                showlegend=True,
                height=800,
                hovermode="x unified",
                template="plotly_white",
            )

            # Update axes
            fig.update_yaxes(title_text="Stock Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volatility", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
            fig.update_xaxes(title_text="Date", row=3, col=1)

            # Add time range selector
            fig.update_layout(
                xaxis3=dict(
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
                                dict(step="all"),
                            ]
                        )
                    ),
                    rangeslider=dict(visible=False),
                    type="date",
                )
            )

            # Save the demo
            output_path = "outputs/mews_demo_dashboard.html"
            fig.write_html(output_path, include_plotlyjs="cdn")

            print(f"\n✅ Demo dashboard created: {output_path}")

            # Show some statistics
            print(f"\n📊 Key Statistics:")
            for symbol in symbols:
                symbol_data = demo_df[demo_df["Symbol"] == symbol]
                if not symbol_data.empty:
                    current_price = symbol_data["Close"].iloc[-1]
                    price_change = (
                        (symbol_data["Close"].iloc[-1] / symbol_data["Close"].iloc[0])
                        - 1
                    ) * 100
                    avg_vol = symbol_data.get("volatility", pd.Series([0])).mean()

                    print(
                        f"   {symbol}: ${current_price:.2f} ({price_change:+.1f}%), Avg Vol: {avg_vol:.4f}"
                    )

            # Show feature categories
            feature_categories = {
                "Price Features": [
                    col
                    for col in df.columns
                    if any(
                        x in col for x in ["Close", "Open", "High", "Low", "SMA", "EMA"]
                    )
                ],
                "Technical Indicators": [
                    col
                    for col in df.columns
                    if any(x in col for x in ["RSI", "MACD", "BB", "volatility"])
                ],
                "Volume Features": [
                    col for col in df.columns if "Volume" in col or "volume" in col
                ],
            }

            print(f"\n🔧 Available Features:")
            for category, features in feature_categories.items():
                print(f"   {category}: {len(features)} features")

            print(f"\n🎯 The MEWS system provides:")
            print(f"   ✅ Multi-asset risk monitoring")
            print(f"   ✅ Technical indicator analysis")
            print(f"   ✅ Volatility tracking")
            print(f"   ✅ Interactive visualizations")
            print(f"   ✅ Real-time risk assessment")

            return output_path

        else:
            print(f"❌ No data found at {data_path}")
            return None

    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    create_demo_visualization()
