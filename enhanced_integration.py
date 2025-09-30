"""
Integration script for enhanced risk analysis
Updates main.py to use the new enhanced features
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging

import pandas as pd

from enhanced_risk_analyzer import EnhancedRiskAnalyzer
from enhanced_risk_timeline import EnhancedRiskTimelineVisualizer


def integrate_enhanced_risk_system(df: pd.DataFrame, symbols: list = None) -> dict:
    """
    Integrate enhanced risk analysis into existing system

    Args:
        df: Original integrated dataset
        symbols: List of symbols to analyze

    Returns:
        Dictionary with enhanced results and visualization paths
    """
    logger = logging.getLogger(__name__)
    logger.info("Running enhanced risk analysis...")

    results = {}

    try:
        # 1. Create enhanced features
        risk_analyzer = EnhancedRiskAnalyzer()
        enhanced_df = risk_analyzer.create_enhanced_features(df)

        # 2. Create composite risk scores
        risk_df = risk_analyzer.create_risk_score(enhanced_df)

        # 3. Create enhanced visualizations
        timeline_viz = EnhancedRiskTimelineVisualizer()

        # Enhanced interactive timeline
        timeline_path = timeline_viz.create_enhanced_risk_timeline(
            df=risk_df,
            symbols=symbols,
            show_confidence_bands=True,
            show_regime_changes=True,
        )

        # Risk summary dashboard
        summary_path = timeline_viz.create_risk_summary_dashboard(
            df=risk_df, symbols=symbols
        )

        results = {
            "enhanced_dataset": risk_df,
            "timeline_path": timeline_path,
            "summary_dashboard_path": summary_path,
            "feature_count": len(
                [col for col in risk_df.columns if col not in df.columns]
            ),
            "symbols_analyzed": (
                len(symbols)
                if symbols
                else len(df["Symbol"].unique()) if "Symbol" in df.columns else 0
            ),
        }

        logger.info(
            f"Enhanced risk analysis completed with {results['feature_count']} new features"
        )

    except Exception as e:
        logger.error(f"Enhanced risk analysis failed: {str(e)}")
        results["error"] = str(e)

    return results


if __name__ == "__main__":
    # Example usage
    print("Enhanced Risk Analysis Integration")
    print("This script integrates enhanced features into your MEWS system")
    print("\nTo use in your main pipeline, import and call:")
    print("from enhanced_integration import integrate_enhanced_risk_system")
    print("results = integrate_enhanced_risk_system(df, symbols)")
