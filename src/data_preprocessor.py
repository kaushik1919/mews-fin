"""
Data preprocessing pipeline
Handles cleaning, normalization, and feature engineering for structured and textual data
"""

import logging
import os
import re
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class DataPreprocessor:
    """Handles all data preprocessing tasks"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.text_processor = TextPreprocessor()

    def preprocess_stock_data(self, stock_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess stock market data

        Args:
            stock_df: Raw stock data DataFrame

        Returns:
            Cleaned and processed stock DataFrame
        """
        self.logger.info("Preprocessing stock data...")

        if stock_df.empty:
            return stock_df

        df = stock_df.copy()

        # Ensure Date column exists and is datetime
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        else:
            df.reset_index(inplace=True)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])

        # Sort by symbol and date
        if "Symbol" in df.columns:
            df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

        # Handle missing values in price data
        price_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        existing_price_cols = [col for col in price_columns if col in df.columns]

        for col in existing_price_cols:
            # Forward fill then backward fill
            df[col] = df.groupby("Symbol")[col].ffill().bfill()

            # If still missing, interpolate
            df[col] = df.groupby("Symbol")[col].transform(lambda x: x.interpolate())

        # Calculate returns if not present
        if "Returns" not in df.columns and "Close" in df.columns:
            df["Returns"] = df.groupby("Symbol")["Close"].pct_change()

        # Calculate volatility measures
        if "Returns" in df.columns:
            # Rolling volatility (annualized)
            for window in [5, 10, 20, 30]:
                df[f"Volatility_{window}d"] = df.groupby("Symbol")["Returns"].transform(
                    lambda x: x.rolling(window=window).std() * np.sqrt(252)
                )

        # Calculate moving averages
        if "Close" in df.columns:
            for window in [5, 10, 20, 50, 100, 200]:
                df[f"MA_{window}"] = df.groupby("Symbol")["Close"].transform(
                    lambda x: x.rolling(window=window).mean()
                )

        # Technical indicators
        df = self._calculate_technical_indicators(df)

        # Handle financial ratios
        ratio_columns = ["PE_Ratio", "Debt_to_Equity", "ROE", "PB_Ratio", "Beta"]
        existing_ratio_cols = [col for col in ratio_columns if col in df.columns]

        for col in existing_ratio_cols:
            # Remove extreme outliers (beyond 99th percentile)
            upper_bound = df[col].quantile(0.99)
            lower_bound = df[col].quantile(0.01)
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])

            # Fill missing values with sector/industry median
            if "Symbol" in df.columns:
                df[col] = df.groupby("Symbol")[col].ffill().bfill()

            # If still missing, use overall median
            df[col] = df[col].fillna(df[col].median())

        # Normalize price data (optional - for ML models)
        df = self._add_normalized_features(df)

        # Add market indicators
        df = self._add_market_indicators(df)

        # Final pass to remove any residual NaNs introduced by rolling windows
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(0.0)

        self.logger.info(
            f"Processed stock data: {len(df)} rows, {len(df.columns)} columns"
        )
        return df

    def preprocess_news_data(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess news data

        Args:
            news_df: Raw news data DataFrame

        Returns:
            Cleaned and processed news DataFrame
        """
        self.logger.info("Preprocessing news data...")

        if news_df.empty:
            return news_df

        df = news_df.copy()

        # Clean and standardize dates
        if "published_date" in df.columns:
            df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
            df = df.dropna(subset=["published_date"])

        # Clean text fields
        text_columns = ["title", "description", "content", "combined_text"]
        existing_text_cols = [col for col in text_columns if col in df.columns]

        for col in existing_text_cols:
            df[col] = df[col].fillna("")
            df[col] = df[col].apply(self.text_processor.clean_text)

        # Create combined text if not exists
        if "combined_text" not in df.columns:
            df["combined_text"] = (
                df.get("title", "")
                + " "
                + df.get("description", "")
                + " "
                + df.get("content", "")
            ).str.strip()

        # Add text features
        df["title_word_count"] = df["title"].str.split().str.len()
        df["description_word_count"] = df["description"].str.split().str.len()
        df["combined_word_count"] = df["combined_text"].str.split().str.len()

        # Filter out very short articles (likely noise)
        min_word_count = 5
        df = df[df["combined_word_count"] >= min_word_count]

        # Add date features
        if "published_date" in df.columns:
            df["year"] = df["published_date"].dt.year
            df["month"] = df["published_date"].dt.month
            df["day_of_week"] = df["published_date"].dt.dayofweek
            df["hour"] = df["published_date"].dt.hour

        # Remove duplicates
        duplicate_cols = (
            ["title", "symbol"]
            if "title" in df.columns and "symbol" in df.columns
            else None
        )
        if duplicate_cols:
            df = df.drop_duplicates(subset=duplicate_cols, keep="first")

        self.logger.info(
            f"Processed news data: {len(df)} rows, {len(df.columns)} columns"
        )
        return df

    def preprocess_sec_data(self, sec_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess SEC filings data

        Args:
            sec_df: Raw SEC data DataFrame

        Returns:
            Cleaned and processed SEC DataFrame
        """
        self.logger.info("Preprocessing SEC filings data...")

        if sec_df.empty:
            return sec_df

        df = sec_df.copy()

        # Clean dates
        if "filing_date" in df.columns:
            df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
            df = df.dropna(subset=["filing_date"])

        # Clean text sections
        text_columns = ["mda_text", "risk_factors_text"]
        existing_text_cols = [col for col in text_columns if col in df.columns]

        for col in existing_text_cols:
            df[col] = df[col].fillna("")
            df[col] = df[col].apply(self.text_processor.clean_sec_text)

        # Add text features
        for col in existing_text_cols:
            base_name = col.replace("_text", "")
            df[f"{base_name}_word_count"] = df[col].str.split().str.len()
            df[f"{base_name}_sentence_count"] = df[col].str.split(".").str.len()
            df[f"{base_name}_paragraph_count"] = df[col].str.split("\n").str.len()

        # Filter out filings with insufficient text
        min_words = 100
        if "mda_text" in df.columns:
            df = df[df["mda_word_count"] >= min_words]

        # Add filing period features
        if "filing_date" in df.columns:
            df["filing_year"] = df["filing_date"].dt.year
            df["filing_quarter"] = df["filing_date"].dt.quarter
            df["filing_month"] = df["filing_date"].dt.month

        # Sort by symbol and filing date
        if all(col in df.columns for col in ["symbol", "filing_date"]):
            df = df.sort_values(["symbol", "filing_date"]).reset_index(drop=True)

        self.logger.info(
            f"Processed SEC data: {len(df)} rows, {len(df.columns)} columns"
        )
        return df

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional technical indicators"""

        if "Close" not in df.columns:
            return df

        # RSI calculation
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        # MACD calculation
        def calculate_macd(prices, fast=12, slow=26, signal=9):
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram

        # Bollinger Bands
        def calculate_bollinger_bands(prices, window=20, num_std=2):
            rolling_mean = prices.rolling(window=window).mean()
            rolling_std = prices.rolling(window=window).std()
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
            return upper_band, lower_band

        # Apply calculations grouped by symbol
        if "Symbol" in df.columns:
            # RSI
            df["RSI"] = df.groupby("Symbol")["Close"].transform(calculate_rsi)

            # MACD
            macd_data = df.groupby("Symbol")["Close"].transform(
                lambda x: calculate_macd(x)[0]
            )
            df["MACD"] = macd_data

            # Bollinger Bands
            def calculate_bb_upper(prices, window=20, num_std=2):
                rolling_mean = prices.rolling(window=window).mean()
                rolling_std = prices.rolling(window=window).std()
                return rolling_mean + (rolling_std * num_std)

            def calculate_bb_lower(prices, window=20, num_std=2):
                rolling_mean = prices.rolling(window=window).mean()
                rolling_std = prices.rolling(window=window).std()
                return rolling_mean - (rolling_std * num_std)

            df["BB_Upper"] = df.groupby("Symbol")["Close"].transform(calculate_bb_upper)
            df["BB_Lower"] = df.groupby("Symbol")["Close"].transform(calculate_bb_lower)
            df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (
                df["BB_Upper"] - df["BB_Lower"]
            )

            # Volume indicators
            if "Volume" in df.columns:
                df["Volume_MA_20"] = df.groupby("Symbol")["Volume"].transform(
                    lambda x: x.rolling(20).mean()
                )
                df["Volume_Ratio"] = df["Volume"] / df["Volume_MA_20"]

        return df

    def _add_normalized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add normalized features for ML models"""

        # Price normalization (min-max scaling by symbol)
        if all(col in df.columns for col in ["Close", "Symbol"]):
            df["Close_Normalized"] = df.groupby("Symbol")["Close"].transform(
                lambda x: (
                    (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
                )
            )

        # Volume normalization
        if all(col in df.columns for col in ["Volume", "Symbol"]):
            df["Volume_Normalized"] = df.groupby("Symbol")["Volume"].transform(
                lambda x: (
                    (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
                )
            )

        # Z-score normalization for ratios
        ratio_cols = ["PE_Ratio", "Debt_to_Equity", "ROE", "Beta"]
        existing_ratios = [col for col in ratio_cols if col in df.columns]

        for col in existing_ratios:
            df[f"{col}_zscore"] = df.groupby("Symbol")[col].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )

        return df

    def _add_market_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market-wide indicators"""

        if "Date" not in df.columns or "Close" not in df.columns:
            return df

        # Market volatility (VIX proxy - average volatility across all stocks)
        daily_volatility = df.groupby("Date")["Returns"].std().reset_index()
        daily_volatility.columns = ["Date", "Market_Volatility"]
        df = df.merge(daily_volatility, on="Date", how="left")

        # Market return (equal-weighted average)
        daily_returns = df.groupby("Date")["Returns"].mean().reset_index()
        daily_returns.columns = ["Date", "Market_Return"]
        df = df.merge(daily_returns, on="Date", how="left")

        # Market trend (percentage of stocks above 50-day MA)
        if "MA_50" in df.columns:
            market_trend = (
                df.groupby("Date")
                .apply(lambda x: (x["Close"] > x["MA_50"]).mean())
                .reset_index()
            )
            market_trend.columns = ["Date", "Market_Trend"]
            df = df.merge(market_trend, on="Date", how="left")

        return df

    def create_risk_labels(
        self,
        df: pd.DataFrame,
        volatility_threshold: float = 0.3,
        return_threshold: float = -0.1,
        window: int = 5,
    ) -> pd.DataFrame:
        """
        Create risk labels based on future market movements

        Args:
            df: Stock data DataFrame
            volatility_threshold: Volatility threshold for high risk
            return_threshold: Return threshold for high risk (negative)
            window: Forward-looking window in days

        Returns:
            DataFrame with risk labels
        """
        self.logger.info("Creating risk labels...")

        if df.empty or "Returns" not in df.columns:
            return df

        df = df.copy()

        # Calculate forward-looking metrics
        if "Symbol" in df.columns:
            # Forward returns
            df["Forward_Return"] = df.groupby("Symbol")["Returns"].shift(-window)

            # Forward volatility
            df["Forward_Volatility"] = (
                df.groupby("Symbol")["Returns"]
                .rolling(window=window)
                .std()
                .shift(-window)
            ).reset_index(0, drop=True)

            valid_mask = df["Forward_Return"].notna() & df["Forward_Volatility"].notna()

            df["Risk_Label"] = 0
            df["Risk_Score"] = 0.0

            if valid_mask.any():
                high_vol_mask = df["Forward_Volatility"] > volatility_threshold
                neg_return_mask = df["Forward_Return"] < return_threshold

                df.loc[valid_mask & (high_vol_mask | neg_return_mask), "Risk_Label"] = 1

                vol_score = (df["Forward_Volatility"] / volatility_threshold).clip(
                    lower=0, upper=2
                ) / 2
                return_score = (-df["Forward_Return"] / -return_threshold).clip(
                    lower=0, upper=2
                ) / 2
                combined_score = ((vol_score + return_score) / 2).clip(0, 1)
                df.loc[valid_mask, "Risk_Score"] = combined_score.loc[
                    valid_mask
                ].fillna(0.0)
                df.loc[~valid_mask, ["Forward_Return", "Forward_Volatility"]] = 0.0
            else:
                self.logger.warning(
                    "Insufficient forward data to compute risk labels for window=%s",
                    window,
                )

        valid_observations = (
            int(df["Risk_Label"].notna().sum()) if "Risk_Label" in df.columns else 0
        )
        high_risk_count = (
            int(df.loc[df["Risk_Label"].notna(), "Risk_Label"].sum())
            if "Risk_Label" in df.columns
            else 0
        )
        self.logger.info(
            "Created risk labels: %s/%s high-risk periods",
            high_risk_count,
            valid_observations,
        )
        return df


class TextPreprocessor:
    """Handles text preprocessing for news and SEC filings"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""  # type: ignore

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'"$%]', " ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def clean_sec_text(self, text: str) -> str:
        """Specialized cleaning for SEC filings"""
        if not isinstance(text, str):
            return ""  # type: ignore

        text = self.clean_text(text)

        # Remove common SEC filing artifacts
        artifacts = [
            r"table of contents",
            r"page \d+",
            r"form 10-[kq]",
            r"item \d+[a-z]*\.",
            r"\.{3,}",
            r"-{3,}",
            r"pursuant to section",
            r"securities exchange act",
            r"commission file number",
        ]

        for pattern in artifacts:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

        # Remove excessive line breaks
        text = re.sub(r"\n+", " ", text)

        # Remove extra whitespace again
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def extract_keywords(self, text: str, top_n: int = 20) -> List[str]:
        """Extract important keywords from text"""
        if not text:
            return []

        # Simple keyword extraction (can be enhanced with TF-IDF or NLP libraries)
        words = text.lower().split()

        # Remove stop words and short words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "this",
            "that",
            "these",
            "those",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "what",
            "which",
            "who",
            "whom",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "having",
            "do",
            "does",
            "did",
            "doing",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "shall",
        }

        # Filter words
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]

        # Count frequency
        from collections import Counter

        word_freq = Counter(keywords)

        # Return top keywords
        return [word for word, freq in word_freq.most_common(top_n)]
