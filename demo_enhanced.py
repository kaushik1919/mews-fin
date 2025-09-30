"""
Demo script to showcase enhanced MEWS features
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from enhanced_risk_analyzer import EnhancedRiskAnalyzer
from enhanced_risk_timeline import EnhancedRiskTimelineVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demo_enhanced_features():
    """Demo the enhanced features with existing data"""

    print("🚀 MEWS Enhanced Risk Analysis Demo")
    print("=" * 50)

    try:
        # Load existing integrated data
        data_path = "data/integrated_dataset.csv"

        if os.path.exists(data_path):
            logger.info(f"Loading existing data from {data_path}")
            df = pd.read_csv(data_path)

            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])

            print(f"📊 Loaded dataset: {len(df)} rows, {len(df.columns)} columns")

            # Get symbols
            symbols = df["Symbol"].unique()[:3] if "Symbol" in df.columns else []
            print(f"🏢 Analyzing symbols: {list(symbols)}")

            # 1. Enhanced Risk Analysis
            print("\n1️⃣ Creating Enhanced Risk Features...")
            risk_analyzer = EnhancedRiskAnalyzer()
            enhanced_df = risk_analyzer.create_enhanced_features(df)

            new_features = len(enhanced_df.columns) - len(df.columns)
            print(f"   ✅ Added {new_features} new features")

            # Show some new features
            new_feature_names = [
                col for col in enhanced_df.columns if col not in df.columns
            ][:10]
            print(f"   📈 Sample new features: {new_feature_names}")

            # 2. Composite Risk Scoring
            print("\n2️⃣ Creating Composite Risk Scores...")
            risk_df = risk_analyzer.create_risk_score(enhanced_df)

            if "composite_risk_score" in risk_df.columns:
                avg_risk = risk_df["composite_risk_score"].mean()
                max_risk = risk_df["composite_risk_score"].max()
                print(f"   ✅ Average risk score: {avg_risk:.3f}")
                print(f"   ⚠️  Maximum risk score: {max_risk:.3f}")

            # 3. Enhanced Visualizations
            print("\n3️⃣ Creating Enhanced Visualizations...")
            timeline_viz = EnhancedRiskTimelineVisualizer()

            # Enhanced timeline
            timeline_path = timeline_viz.create_enhanced_risk_timeline(
                df=risk_df,
                symbols=symbols,
                show_confidence_bands=False,  # Skip confidence bands for demo
                show_regime_changes=True,
            )

            if timeline_path:
                print(f"   📊 Enhanced timeline: {timeline_path}")

            # Summary dashboard
            summary_path = timeline_viz.create_risk_summary_dashboard(
                df=risk_df, symbols=symbols
            )

            if summary_path:
                print(f"   📋 Summary dashboard: {summary_path}")

            # 4. Risk Analysis Summary
            print("\n4️⃣ Risk Analysis Summary")
            print("-" * 30)

            for symbol in symbols:
                symbol_data = risk_df[risk_df["Symbol"] == symbol]
                if (
                    not symbol_data.empty
                    and "composite_risk_score" in symbol_data.columns
                ):
                    latest_risk = symbol_data["composite_risk_score"].iloc[-1]
                    avg_risk = symbol_data["composite_risk_score"].mean()

                    risk_level = (
                        "🟢 LOW"
                        if latest_risk < 0.3
                        else "🟡 MEDIUM" if latest_risk < 0.7 else "🔴 HIGH"
                    )

                    print(
                        f"   {symbol}: {risk_level} (Current: {latest_risk:.3f}, Avg: {avg_risk:.3f})"
                    )

            print("\n🎉 Enhanced MEWS Demo Complete!")
            print(f"\n📁 Check your outputs folder for:")
            print(f"   - Enhanced timeline: enhanced_risk_timeline_interactive.html")
            print(f"   - Summary dashboard: risk_summary_dashboard.html")
            print(f"   - Original timeline: risk_timeline_interactive.html")

        else:
            print(f"❌ No data found at {data_path}")
            print("   Please run: python main.py --full-pipeline first")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"❌ Demo failed: {str(e)}")


if __name__ == "__main__":
    demo_enhanced_features()
