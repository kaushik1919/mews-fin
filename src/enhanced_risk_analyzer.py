"""
Enhanced Risk Analyzer
Combines sentiment analysis with structured data for improved risk prediction
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class EnhancedRiskAnalyzer:
    """Advanced risk analysis combining multiple data sources with improved feature engineering"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create enhanced features combining sentiment and structured data

        Args:
            df: DataFrame with stock data and sentiment scores

        Returns:
            DataFrame with enhanced features
        """
        self.logger.info("Creating enhanced risk features...")

        enhanced_df = df.copy()

        # 1. Advanced Sentiment Features
        enhanced_df = self._create_sentiment_features(enhanced_df)

        # 2. Technical Indicator Enhancements
        enhanced_df = self._create_technical_features(enhanced_df)

        # 3. Market Regime Features
        enhanced_df = self._create_regime_features(enhanced_df)

        # 4. Cross-Asset Features
        enhanced_df = self._create_cross_asset_features(enhanced_df)

        # 5. Temporal Features
        enhanced_df = self._create_temporal_features(enhanced_df)

        return enhanced_df

    def _create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced sentiment-based features"""

        # Find available sentiment columns
        sentiment_cols = [col for col in df.columns if "sentiment" in col.lower()]
        news_sentiment_col = None
        sec_sentiment_col = None

        # Try to identify news and SEC sentiment columns
        for col in sentiment_cols:
            if "news" in col.lower() or "title" in col.lower():
                news_sentiment_col = col
            elif "sec" in col.lower() or "filing" in col.lower():
                sec_sentiment_col = col

        # If no specific columns found, use any sentiment column
        if not news_sentiment_col and sentiment_cols:
            news_sentiment_col = sentiment_cols[0]

        # Sentiment momentum (rate of change)
        if news_sentiment_col and news_sentiment_col in df.columns:
            df["sentiment_momentum_1d"] = df.groupby("Symbol")[
                news_sentiment_col
            ].pct_change(1)
            df["sentiment_momentum_3d"] = df.groupby("Symbol")[
                news_sentiment_col
            ].pct_change(3)
            df["sentiment_momentum_7d"] = df.groupby("Symbol")[
                news_sentiment_col
            ].pct_change(7)

            # Sentiment volatility
            df["sentiment_volatility_7d"] = (
                df.groupby("Symbol")[news_sentiment_col]
                .rolling(7)
                .std()
                .reset_index(0, drop=True)
            )
            df["sentiment_volatility_30d"] = (
                df.groupby("Symbol")[news_sentiment_col]
                .rolling(30)
                .std()
                .reset_index(0, drop=True)
            )

            # Sentiment extremes
            df["sentiment_extreme_positive"] = (
                df[news_sentiment_col]
                > df.groupby("Symbol")[news_sentiment_col]
                .rolling(30)
                .quantile(0.9)
                .reset_index(0, drop=True)
            ).astype(int)
            df["sentiment_extreme_negative"] = (
                df[news_sentiment_col]
                < df.groupby("Symbol")[news_sentiment_col]
                .rolling(30)
                .quantile(0.1)
                .reset_index(0, drop=True)
            ).astype(int)

        # SEC filing sentiment features
        if sec_sentiment_col and sec_sentiment_col in df.columns:
            df["sec_sentiment_trend_30d"] = (
                df.groupby("Symbol")[sec_sentiment_col]
                .rolling(30)
                .apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
                .reset_index(0, drop=True)
            )

            # Combined sentiment score with weighting
            if news_sentiment_col and news_sentiment_col in df.columns:
                df["combined_sentiment_weighted"] = (
                    0.7 * df[news_sentiment_col] + 0.3 * df[sec_sentiment_col]
                )

                # Sentiment divergence (when news and SEC sentiment disagree)
                df["sentiment_divergence"] = abs(
                    df[news_sentiment_col] - df[sec_sentiment_col]
                )

        return df

    def _create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced technical indicator features"""

        # Advanced volatility measures
        if "Close" in df.columns:
            # Parkinson volatility (uses high-low range)
            if all(col in df.columns for col in ["High", "Low"]):
                df["parkinson_volatility"] = (
                    df.groupby("Symbol")
                    .apply(
                        lambda x: np.sqrt(
                            252 * (np.log(x["High"] / x["Low"]) ** 2).rolling(20).mean()
                        )
                    )
                    .reset_index(0, drop=True)
                )

            # Price acceleration (second derivative)
            df["price_acceleration"] = (
                df.groupby("Symbol")["Close"].pct_change().pct_change()
            )

            # Support and resistance levels
            df["resistance_level"] = (
                df.groupby("Symbol")["High"].rolling(20).max().reset_index(0, drop=True)
            )
            df["support_level"] = (
                df.groupby("Symbol")["Low"].rolling(20).min().reset_index(0, drop=True)
            )
            df["price_near_resistance"] = (
                df["Close"] / df["resistance_level"] > 0.98
            ).astype(int)
            df["price_near_support"] = (
                df["Close"] / df["support_level"] < 1.02
            ).astype(int)

        # Volume-price divergence
        if all(col in df.columns for col in ["Close", "Volume"]):
            # On-balance volume
            df["price_change"] = df.groupby("Symbol")["Close"].pct_change()
            df["obv_direction"] = np.where(
                df["price_change"] > 0, df["Volume"], -df["Volume"]
            )
            df["obv"] = df.groupby("Symbol")["obv_direction"].cumsum()

            # Volume-weighted average price deviation
            df["vwap"] = (
                df.groupby("Symbol")
                .apply(
                    lambda x: (x["Close"] * x["Volume"]).rolling(20).sum()
                    / x["Volume"].rolling(20).sum()
                )
                .reset_index(0, drop=True)
            )
            df["price_vwap_deviation"] = (df["Close"] - df["vwap"]) / df["vwap"]

        return df

    def _create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market regime identification features"""

        if "Close" in df.columns:
            # Trend strength
            df["trend_strength"] = (
                df.groupby("Symbol")["Close"]
                .rolling(20)
                .apply(
                    lambda x: (
                        abs(np.polyfit(range(len(x)), x, 1)[0]) if len(x) > 1 else 0
                    )
                )
                .reset_index(0, drop=True)
            )

            # Market phase identification
            df["sma_20"] = (
                df.groupby("Symbol")["Close"]
                .rolling(20)
                .mean()
                .reset_index(0, drop=True)
            )
            df["sma_50"] = (
                df.groupby("Symbol")["Close"]
                .rolling(50)
                .mean()
                .reset_index(0, drop=True)
            )
            df["sma_200"] = (
                df.groupby("Symbol")["Close"]
                .rolling(200)
                .mean()
                .reset_index(0, drop=True)
            )

            # Bull/Bear market indicators
            df["bull_market"] = (
                (df["sma_20"] > df["sma_50"]) & (df["sma_50"] > df["sma_200"])
            ).astype(int)
            df["bear_market"] = (
                (df["sma_20"] < df["sma_50"]) & (df["sma_50"] < df["sma_200"])
            ).astype(int)

            # Volatility regime
            df["returns"] = df.groupby("Symbol")["Close"].pct_change()
            df["volatility_20d"] = (
                df.groupby("Symbol")["returns"]
                .rolling(20)
                .std()
                .reset_index(0, drop=True)
            )
            df["volatility_regime"] = (
                df.groupby("Symbol")["volatility_20d"]
                .rolling(60)
                .rank(pct=True)
                .reset_index(0, drop=True)
            )
            df["high_volatility_regime"] = (df["volatility_regime"] > 0.8).astype(int)
            df["low_volatility_regime"] = (df["volatility_regime"] < 0.2).astype(int)

        return df

    def _create_cross_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on cross-asset relationships"""

        # Find available sentiment columns again
        sentiment_cols = [col for col in df.columns if "sentiment" in col.lower()]
        news_sentiment_col = None

        for col in sentiment_cols:
            if "news" in col.lower() or "title" in col.lower():
                news_sentiment_col = col
                break

        if not news_sentiment_col and sentiment_cols:
            news_sentiment_col = sentiment_cols[0]

        if (
            "Symbol" in df.columns
            and len(df["Symbol"].unique()) > 1
            and news_sentiment_col
        ):
            # Market-wide sentiment average
            market_sentiment = (
                df.groupby("Date")[news_sentiment_col].mean().reset_index()
            )
            market_sentiment.columns = ["Date", "market_sentiment_avg"]
            df = df.merge(market_sentiment, on="Date", how="left")

            # Relative sentiment (vs market)
            if "market_sentiment_avg" in df.columns:
                df["relative_sentiment"] = (
                    df[news_sentiment_col] - df["market_sentiment_avg"]
                )

            # Sector-specific features (if sector info available)
            # This would require additional sector mapping

        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""

        if "Date" in df.columns:
            try:
                # Ensure Date column is datetime
                if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

                # Remove any rows with invalid dates
                df = df.dropna(subset=["Date"])

                if df.empty:
                    return df

                # Day of week effect
                df["day_of_week"] = df["Date"].dt.dayofweek
                df["is_monday"] = (df["day_of_week"] == 0).astype(int)
                df["is_friday"] = (df["day_of_week"] == 4).astype(int)

                # Month effect
                df["month"] = df["Date"].dt.month
                df["is_january"] = (df["month"] == 1).astype(int)
                df["is_december"] = (df["month"] == 12).astype(int)

                # Earnings season approximation (rough quarters)
                df["earnings_season"] = df["month"].isin([1, 4, 7, 10]).astype(int)

                # Days to month end
                df["days_to_month_end"] = (
                    df["Date"].dt.days_in_month - df["Date"].dt.day
                )
                df["end_of_month"] = (df["days_to_month_end"] <= 3).astype(int)

            except Exception as e:
                print(f"Warning: Could not create time-based features: {str(e)}")
                # Continue without time-based features

        return df

    def create_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite risk score combining multiple factors

        Args:
            df: DataFrame with enhanced features

        Returns:
            DataFrame with composite risk scores
        """
        self.logger.info("Creating composite risk scores...")

        risk_df = df.copy()

        # Initialize risk components
        sentiment_risk = 0
        technical_risk = 0
        volatility_risk = 0
        volume_risk = 0

        # Sentiment risk component
        if "sentiment_extreme_negative" in risk_df.columns:
            sentiment_risk += risk_df["sentiment_extreme_negative"] * 0.3
        if "sentiment_divergence" in risk_df.columns:
            sentiment_risk += (
                risk_df["sentiment_divergence"]
                > risk_df["sentiment_divergence"].quantile(0.8)
            ).astype(int) * 0.2
        if "sentiment_momentum_1d" in risk_df.columns:
            sentiment_risk += (risk_df["sentiment_momentum_1d"] < -0.1).astype(
                int
            ) * 0.3
        if "relative_sentiment" in risk_df.columns:
            sentiment_risk += (risk_df["relative_sentiment"] < -0.2).astype(int) * 0.2

        # Technical risk component
        if "price_near_support" in risk_df.columns:
            technical_risk += risk_df["price_near_support"] * 0.3
        if "bear_market" in risk_df.columns:
            technical_risk += risk_df["bear_market"] * 0.4
        if "price_acceleration" in risk_df.columns:
            technical_risk += (risk_df["price_acceleration"] < -0.01).astype(int) * 0.3

        # Volatility risk component
        if "high_volatility_regime" in risk_df.columns:
            volatility_risk += risk_df["high_volatility_regime"] * 0.5
        if "parkinson_volatility" in risk_df.columns:
            volatility_risk += (
                risk_df["parkinson_volatility"]
                > risk_df["parkinson_volatility"].quantile(0.9)
            ).astype(int) * 0.5

        # Volume risk component
        if "price_vwap_deviation" in risk_df.columns:
            volume_risk += (abs(risk_df["price_vwap_deviation"]) > 0.05).astype(
                int
            ) * 0.5

        # Composite risk score (weighted combination)
        risk_df["sentiment_risk_score"] = sentiment_risk
        risk_df["technical_risk_score"] = technical_risk
        risk_df["volatility_risk_score"] = volatility_risk
        risk_df["volume_risk_score"] = volume_risk

        # Overall composite risk score
        risk_df["composite_risk_score"] = (
            0.3 * sentiment_risk
            + 0.3 * technical_risk
            + 0.25 * volatility_risk
            + 0.15 * volume_risk
        )

        # Normalize to 0-1 scale
        risk_df["composite_risk_score"] = np.clip(risk_df["composite_risk_score"], 0, 1)

        return risk_df
