"""
Data integration module
Merges structured and unstructured features indexed by date and company
"""

import logging
import os
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class DataIntegrator:
    """Integrates all data sources into a unified dataset for modeling"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def integrate_all_data(
        self,
        stock_df: pd.DataFrame,
        news_df: pd.DataFrame = None,
        sec_df: pd.DataFrame = None,
        sentiment_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Integrate all data sources into a unified dataset

        Args:
            stock_df: Stock price and financial data
            news_df: News data with sentiment (optional)
            sec_df: SEC filings data with sentiment (optional)
            sentiment_df: Aggregated sentiment data (optional)

        Returns:
            Integrated DataFrame ready for modeling
        """
        self.logger.info("Starting data integration process...")

        if stock_df.empty:
            self.logger.error("Stock data is empty - cannot proceed with integration")
            return pd.DataFrame()

        # Start with stock data as base
        integrated_df = stock_df.copy()

        # Ensure proper date format
        integrated_df = self._standardize_dates(integrated_df)

        # Add sentiment features
        if sentiment_df is not None and not sentiment_df.empty:
            integrated_df = self._merge_sentiment_data(integrated_df, sentiment_df)

        # Add news features
        if news_df is not None and not news_df.empty:
            integrated_df = self._merge_news_features(integrated_df, news_df)

        # Add SEC filing features
        if sec_df is not None and not sec_df.empty:
            integrated_df = self._merge_sec_features(integrated_df, sec_df)

        # Add derived features
        integrated_df = self._add_derived_features(integrated_df)

        # Add lag features
        integrated_df = self._add_lag_features(integrated_df)

        # Add market regime indicators
        integrated_df = self._add_market_regime_indicators(integrated_df)

        # Clean final dataset
        integrated_df = self._clean_integrated_data(integrated_df)

        self.logger.info(
            f"Data integration completed: {len(integrated_df)} rows, {len(integrated_df.columns)} columns"
        )

        return integrated_df

    def _standardize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize date columns across all datasets"""

        # Convert Date column to datetime (handle timezone-aware datetimes)
        if "Date" in df.columns:
            # Handle timezone-aware datetimes by converting to UTC first if needed
            try:
                df["Date"] = pd.to_datetime(df["Date"])
            except ValueError:
                # If timezone conversion fails, convert to UTC first
                df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
        elif df.index.name == "Date" or hasattr(df.index, "date"):
            df.reset_index(inplace=True)
            try:
                df["Date"] = pd.to_datetime(df["Date"])
            except ValueError:
                df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)

        # Add date components for modeling
        if "Date" in df.columns:
            df["year"] = df["Date"].dt.year
            df["month"] = df["Date"].dt.month
            df["quarter"] = df["Date"].dt.quarter
            df["day_of_year"] = df["Date"].dt.dayofyear
            df["week_of_year"] = df["Date"].dt.isocalendar().week
            df["day_of_week"] = df["Date"].dt.dayofweek
            df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
            df["is_quarter_end"] = df["Date"].dt.is_quarter_end.astype(int)

        return df

    def _merge_sentiment_data(
        self, stock_df: pd.DataFrame, sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge aggregated sentiment data with stock data"""

        self.logger.info("Merging sentiment data...")

        # Prepare sentiment data for merging
        sentiment_clean = sentiment_df.copy()

        if "date" in sentiment_clean.columns:
            sentiment_clean["Date"] = pd.to_datetime(sentiment_clean["date"])

            # Handle timezone compatibility between stock and sentiment data
            if hasattr(stock_df["Date"].dt, "tz"):
                stock_tz = stock_df["Date"].dt.tz
                sentiment_tz = getattr(sentiment_clean["Date"].dt, "tz", None)

                if stock_tz is None and sentiment_tz is not None:
                    # Stock is timezone-naive, sentiment is timezone-aware -> make sentiment naive
                    sentiment_clean["Date"] = sentiment_clean["Date"].dt.tz_localize(
                        None
                    )
                elif stock_tz is not None and sentiment_tz is None:
                    # Stock is timezone-aware, sentiment is naive -> localize sentiment
                    sentiment_clean["Date"] = (
                        sentiment_clean["Date"]
                        .dt.tz_localize("UTC")
                        .dt.tz_convert(stock_tz)
                    )
                elif (
                    stock_tz is not None
                    and sentiment_tz is not None
                    and stock_tz != sentiment_tz
                ):
                    # Both timezone-aware but different -> convert sentiment to stock timezone
                    sentiment_clean["Date"] = sentiment_clean["Date"].dt.tz_convert(
                        stock_tz
                    )

            sentiment_clean.drop("date", axis=1, inplace=True)

        # Merge on Symbol and Date
        merge_cols = ["Symbol", "Date"]
        available_merge_cols = [
            col
            for col in merge_cols
            if col in stock_df.columns and col in sentiment_clean.columns
        ]

        if available_merge_cols:
            merged_df = stock_df.merge(
                sentiment_clean, on=available_merge_cols, how="left"
            )

            # Forward fill sentiment data (sentiment persists for a few days)
            sentiment_cols = [
                col
                for col in sentiment_clean.columns
                if col not in available_merge_cols
            ]
            for col in sentiment_cols:
                if col in merged_df.columns:
                    merged_df[col] = merged_df.groupby("Symbol")[col].ffill()

            self.logger.info(
                f"Merged sentiment data: added {len(sentiment_cols)} sentiment features"
            )
            return merged_df

        return stock_df

    def _merge_news_features(
        self, stock_df: pd.DataFrame, news_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge news-based features with stock data"""

        self.logger.info("Merging news features...")

        news_clean = news_df.copy()

        # Prepare news data
        if "published_date" in news_clean.columns:
            news_clean["Date"] = news_clean["published_date"].dt.date
            news_clean["Date"] = pd.to_datetime(news_clean["Date"])

        # Create daily news aggregations
        if all(col in news_clean.columns for col in ["Symbol", "Date"]):
            news_agg = (
                news_clean.groupby(["Symbol", "Date"])
                .agg(
                    {
                        "combined_sentiment_compound": ["mean", "std", "count"],
                        "sentiment_intensity": "mean",
                        "title_word_count": "mean",
                        "description_word_count": "mean",
                    }
                )
                .reset_index()
            )

            # Flatten column names
            news_agg.columns = [
                "Symbol",
                "Date",
                "news_sentiment_mean",
                "news_sentiment_std",
                "news_count",
                "news_sentiment_intensity",
                "news_title_length_avg",
                "news_desc_length_avg",
            ]

            # Fill NaN values
            news_agg["news_sentiment_std"] = news_agg["news_sentiment_std"].fillna(0)

            # Merge with stock data
            merged_df = stock_df.merge(news_agg, on=["Symbol", "Date"], how="left")

            # Forward fill news features (news effect persists)
            news_cols = [
                "news_sentiment_mean",
                "news_sentiment_std",
                "news_sentiment_intensity",
            ]
            for col in news_cols:
                merged_df[col] = merged_df.groupby("Symbol")[col].ffill()

            # Fill remaining NaN with neutral values
            merged_df["news_count"] = merged_df["news_count"].fillna(0)
            merged_df["news_sentiment_mean"] = merged_df["news_sentiment_mean"].fillna(
                0
            )
            merged_df["news_sentiment_std"] = merged_df["news_sentiment_std"].fillna(0)
            merged_df["news_sentiment_intensity"] = merged_df[
                "news_sentiment_intensity"
            ].fillna(0)

            self.logger.info("Merged news features successfully")
            return merged_df

        return stock_df

    def _merge_sec_features(
        self, stock_df: pd.DataFrame, sec_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge SEC filing features with stock data"""

        self.logger.info("Merging SEC features...")

        sec_clean = sec_df.copy()

        # Prepare SEC data
        if "filing_date" in sec_clean.columns:
            sec_clean["Date"] = pd.to_datetime(sec_clean["filing_date"])

        # SEC filings are quarterly - need to propagate to daily data
        if all(col in sec_clean.columns for col in ["Symbol", "Date"]):

            # Select relevant SEC features
            sec_features = [
                "Symbol",
                "Date",
                "mda_sentiment",
                "risk_factors_sentiment",
                "combined_sec_sentiment",
                "mda_word_count",
                "risk_factors_word_count",
            ]
            existing_features = [
                col for col in sec_features if col in sec_clean.columns
            ]

            sec_subset = sec_clean[existing_features].copy()

            # For each stock, merge with the most recent SEC filing
            merged_df = stock_df.copy()

            for symbol in merged_df["Symbol"].unique():
                stock_symbol_data = merged_df[merged_df["Symbol"] == symbol].copy()
                sec_symbol_data = sec_subset[sec_subset["Symbol"] == symbol].copy()

                if not sec_symbol_data.empty:
                    # Sort SEC data by date
                    sec_symbol_data = sec_symbol_data.sort_values("Date")

                    # Merge as of date (use most recent filing for each stock date)
                    merged_symbol = pd.merge_asof(
                        stock_symbol_data.sort_values("Date"),
                        sec_symbol_data,
                        on="Date",
                        by="Symbol",
                        direction="backward",
                        suffixes=("", "_sec"),
                    )

                    # Update the main dataframe
                    merged_df.update(merged_symbol)

            self.logger.info("Merged SEC features successfully")
            return merged_df

        return stock_df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features combining multiple data sources"""

        # Sentiment-Price momentum features
        if all(col in df.columns for col in ["news_sentiment_mean", "Returns"]):
            df["sentiment_return_correlation"] = (
                df.groupby("Symbol")
                .apply(
                    lambda x: x["news_sentiment_mean"].rolling(20).corr(x["Returns"])
                )
                .reset_index(level=0, drop=True)
            )

        # Sentiment divergence (when sentiment and price move in opposite directions)
        if all(col in df.columns for col in ["news_sentiment_mean", "Price_Change_1d"]):
            df["sentiment_price_divergence"] = (
                (df["news_sentiment_mean"] > 0) & (df["Price_Change_1d"] < 0)
            ).astype(int) + (
                (df["news_sentiment_mean"] < 0) & (df["Price_Change_1d"] > 0)
            ).astype(
                int
            )

        # News volume vs volatility
        if all(col in df.columns for col in ["news_count", "Volatility_20d"]):
            df["news_volatility_ratio"] = df["news_count"] / (
                df["Volatility_20d"] + 1e-6
            )

        # Fundamental-Sentiment composite
        if all(col in df.columns for col in ["PE_Ratio", "news_sentiment_mean"]):
            # High PE with negative sentiment = higher risk
            df["pe_sentiment_risk"] = (df["PE_Ratio"] > df["PE_Ratio"].median()) & (
                df["news_sentiment_mean"] < -0.1
            )
            df["pe_sentiment_risk"] = df["pe_sentiment_risk"].astype(int)

        # SEC-News sentiment alignment
        if all(col in df.columns for col in ["mda_sentiment", "news_sentiment_mean"]):
            df["sec_news_sentiment_diff"] = abs(
                df["mda_sentiment"] - df["news_sentiment_mean"]
            )
            df["sentiment_alignment"] = (
                df["sec_news_sentiment_diff"] < 0.2
            )  # Aligned if difference < 0.2
            df["sentiment_alignment"] = df["sentiment_alignment"].astype(int)

        # Market cap weighted features
        if "Market_Cap" in df.columns:
            # Add market cap deciles for relative analysis
            df["market_cap_decile"] = pd.qcut(
                df["Market_Cap"], 10, labels=False, duplicates="drop"
            )

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features for time series analysis"""

        # Define features to lag
        lag_features = [
            "Returns",
            "Volatility_20d",
            "news_sentiment_mean",
            "news_count",
            "Volume_Ratio",
            "RSI",
            "PE_Ratio",
        ]

        existing_lag_features = [col for col in lag_features if col in df.columns]

        # Add lags for 1, 3, 7, and 14 days
        lag_periods = [1, 3, 7, 14]

        for feature in existing_lag_features:
            for lag in lag_periods:
                df[f"{feature}_lag_{lag}"] = df.groupby("Symbol")[feature].shift(lag)

        # Add rolling means for key features
        rolling_windows = [7, 14, 30]

        for feature in existing_lag_features:
            for window in rolling_windows:
                df[f"{feature}_rolling_mean_{window}"] = (
                    df.groupby("Symbol")[feature]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )

        return df

    def _add_market_regime_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime indicators (bull/bear market, high/low volatility)"""

        if "Market_Return" not in df.columns:
            return df

        # Bull/Bear market indicator (based on 200-day MA of market returns)
        df["market_return_ma200"] = df["Market_Return"].rolling(200).mean()
        df["bull_market"] = (df["Market_Return"] > df["market_return_ma200"]).astype(
            int
        )

        # Volatility regime
        if "Market_Volatility" in df.columns:
            vol_median = df["Market_Volatility"].median()
            df["high_volatility_regime"] = (
                df["Market_Volatility"] > vol_median * 1.5
            ).astype(int)

        # Crisis periods (based on multiple indicators)
        crisis_indicators = []

        # High volatility
        if "Market_Volatility" in df.columns:
            high_vol = df["Market_Volatility"] > df["Market_Volatility"].quantile(0.95)
            crisis_indicators.append(high_vol)

        # Large negative returns
        large_neg_returns = df["Market_Return"] < df["Market_Return"].quantile(0.05)
        crisis_indicators.append(large_neg_returns)

        # High news volume
        if "news_count" in df.columns:
            high_news = df.groupby("Date")["news_count"].transform("sum") > df.groupby(
                "Date"
            )["news_count"].transform("sum").quantile(0.95)
            crisis_indicators.append(high_news)

        # Combine crisis indicators
        if crisis_indicators:
            df["crisis_period"] = sum(crisis_indicators) >= 2
            df["crisis_period"] = df["crisis_period"].astype(int)

        return df

    def _clean_integrated_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleaning of integrated dataset"""

        # Remove columns with too many missing values (> 50%)
        threshold = len(df) * 0.5
        df = df.dropna(axis=1, thresh=threshold)

        # Fill remaining missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != "Date":  # Don't fill Date column
                # Forward fill then backward fill
                df[col] = df.groupby("Symbol")[col].ffill().bfill()

                # If still missing, fill with median
                df[col] = df[col].fillna(df[col].median())

        # Remove infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Fill any remaining NaN with 0 for specific columns
        fill_zero_cols = [
            col
            for col in df.columns
            if any(
                keyword in col.lower()
                for keyword in ["sentiment", "news", "count", "ratio", "volume"]
            )
        ]

        for col in fill_zero_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Sort by Symbol and Date
        if all(col in df.columns for col in ["Symbol", "Date"]):
            df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

        return df

    def create_feature_groups(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Organize features into logical groups for analysis

        Args:
            df: Integrated DataFrame

        Returns:
            Dictionary with feature groups
        """
        feature_groups = {
            "price_features": [],
            "volume_features": [],
            "technical_indicators": [],
            "fundamental_ratios": [],
            "sentiment_features": [],
            "news_features": [],
            "sec_features": [],
            "market_features": [],
            "lag_features": [],
            "derived_features": [],
            "date_features": [],
        }

        for col in df.columns:
            col_lower = col.lower()

            if any(
                keyword in col_lower
                for keyword in ["close", "open", "high", "low", "price", "return"]
            ):
                feature_groups["price_features"].append(col)
            elif any(keyword in col_lower for keyword in ["volume"]):
                feature_groups["volume_features"].append(col)
            elif any(
                keyword in col_lower
                for keyword in ["rsi", "macd", "ma_", "bb_", "volatility"]
            ):
                feature_groups["technical_indicators"].append(col)
            elif any(
                keyword in col_lower
                for keyword in ["pe_ratio", "roe", "debt", "beta", "pb_ratio"]
            ):
                feature_groups["fundamental_ratios"].append(col)
            elif (
                any(keyword in col_lower for keyword in ["sentiment"])
                and "news" not in col_lower
            ):
                feature_groups["sentiment_features"].append(col)
            elif any(keyword in col_lower for keyword in ["news"]):
                feature_groups["news_features"].append(col)
            elif any(
                keyword in col_lower for keyword in ["mda", "sec", "risk_factors"]
            ):
                feature_groups["sec_features"].append(col)
            elif any(keyword in col_lower for keyword in ["market"]):
                feature_groups["market_features"].append(col)
            elif any(keyword in col_lower for keyword in ["lag_", "rolling_mean"]):
                feature_groups["lag_features"].append(col)
            elif any(
                keyword in col_lower
                for keyword in ["divergence", "alignment", "regime", "crisis"]
            ):
                feature_groups["derived_features"].append(col)
            elif any(
                keyword in col_lower
                for keyword in ["year", "month", "quarter", "day", "week"]
            ):
                feature_groups["date_features"].append(col)

        # Log feature group sizes
        for group_name, features in feature_groups.items():
            if features:
                self.logger.info(f"{group_name}: {len(features)} features")

        return feature_groups

    def save_integrated_data(
        self, df: pd.DataFrame, feature_groups: Dict[str, List[str]], output_dir: str
    ):
        """Save integrated dataset and feature information"""

        os.makedirs(output_dir, exist_ok=True)

        # Save main integrated dataset
        main_path = os.path.join(output_dir, "integrated_dataset.csv")
        df.to_csv(main_path, index=False)

        # Save feature groups as JSON
        feature_groups_path = os.path.join(output_dir, "feature_groups.json")
        import json

        with open(feature_groups_path, "w") as f:
            json.dump(feature_groups, f, indent=2)

        # Save dataset summary
        summary_path = os.path.join(output_dir, "dataset_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Integrated Dataset Summary\n")
            f.write(f"========================\n\n")
            f.write(f"Total rows: {len(df)}\n")
            f.write(f"Total columns: {len(df.columns)}\n")
            f.write(
                f"Unique symbols: {df['Symbol'].nunique() if 'Symbol' in df.columns else 'N/A'}\n"
            )

            if "Date" in df.columns:
                f.write(f"Date range: {df['Date'].min()} to {df['Date'].max()}\n")

            f.write(f"\nFeature Groups:\n")
            for group_name, features in feature_groups.items():
                f.write(f"  {group_name}: {len(features)} features\n")

            f.write(f"\nMissing Value Summary:\n")
            missing_summary = df.isnull().sum()
            missing_pct = (missing_summary / len(df) * 100).round(2)
            for col in missing_summary[missing_summary > 0].index:
                f.write(f"  {col}: {missing_summary[col]} ({missing_pct[col]}%)\n")

        self.logger.info(f"Saved integrated dataset to {output_dir}")
        self.logger.info(f"Dataset shape: {df.shape}")
