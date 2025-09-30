"""
Sentiment analysis module
Applies VADER sentiment to news and FinBERT sentiment to SEC filings
"""

import logging
import os
import re
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class SentimentAnalyzer:
    """Handles sentiment analysis for news and SEC filings"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.vader_analyzer = None
        self.finbert_model = None
        self.finbert_tokenizer = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize sentiment analysis models"""
        try:
            # Initialize VADER (for news sentiment)
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self.vader_analyzer = SentimentIntensityAnalyzer()
            self.logger.info("VADER sentiment analyzer initialized")

        except ImportError:
            self.logger.warning("VADER sentiment library not available")

        try:
            # Initialize FinBERT (for financial text sentiment)
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            # Use ProsusAI FinBERT model (free and good for financial text)
            model_name = "ProsusAI/finbert"
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            )

            # Set to evaluation mode
            self.finbert_model.eval()

            self.logger.info("FinBERT model initialized")

        except ImportError:
            self.logger.warning(
                "Transformers library not available - FinBERT sentiment disabled"
            )
        except Exception as e:
            self.logger.warning(f"FinBERT initialization failed: {str(e)}")

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment for a single text snippet with graceful fallbacks."""

        if not text or not str(text).strip():
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

        text_str = str(text)

        if self.vader_analyzer is not None:
            try:
                return self.vader_analyzer.polarity_scores(text_str)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning("VADER sentiment failed, using fallback: %s", exc)

        return self._heuristic_sentiment(text_str)

    def _heuristic_sentiment(self, text: str) -> Dict[str, float]:
        """Lightweight rule-based sentiment fallback when VADER isn't available."""

        tokens = re.findall(r"[A-Za-z']+", text.lower())
        if not tokens:
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

        positive_words = {
            "great",
            "excellent",
            "good",
            "up",
            "gain",
            "bullish",
            "strong",
            "positive",
            "surge",
            "growth",
            "improve",
            "optimistic",
        }
        negative_words = {
            "bad",
            "terrible",
            "loss",
            "crash",
            "down",
            "bearish",
            "weak",
            "negative",
            "drop",
            "decline",
            "risk",
            "volatile",
        }

        def count_hits(words: Iterable[str]) -> int:
            hits = 0
            for token in tokens:
                for lex in words:
                    if token == lex or token.startswith(lex):
                        hits += 1
                        break
            return hits

        pos_hits = count_hits(positive_words)
        neg_hits = count_hits(negative_words)

        if pos_hits == 0 and neg_hits == 0:
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

        raw = (pos_hits - neg_hits) / max(pos_hits + neg_hits, 1)
        compound = float(max(min(raw, 1.0), -1.0))
        pos_score = max(compound, 0.0)
        neg_score = max(-compound, 0.0)
        neu_score = max(0.0, 1.0 - (pos_score + neg_score))

        return {"neg": neg_score, "neu": neu_score, "pos": pos_score, "compound": compound}

    def analyze_news_sentiment(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment of news articles using VADER

        Args:
            news_df: DataFrame with news data

        Returns:
            DataFrame with sentiment scores added
        """
        self.logger.info("Analyzing news sentiment with VADER...")

        if news_df.empty or not self.vader_analyzer:
            return news_df

        df = news_df.copy()

        # Initialize sentiment columns
        df["title_sentiment"] = 0.0
        df["title_sentiment_compound"] = 0.0
        df["description_sentiment"] = 0.0
        df["description_sentiment_compound"] = 0.0
        df["combined_sentiment"] = 0.0
        df["combined_sentiment_compound"] = 0.0

        # Analyze sentiment for different text fields
        text_fields = [
            ("title", "title_sentiment", "title_sentiment_compound"),
            ("description", "description_sentiment", "description_sentiment_compound"),
            ("combined_text", "combined_sentiment", "combined_sentiment_compound"),
        ]

        for text_col, sentiment_col, compound_col in text_fields:
            if text_col in df.columns:
                self.logger.info(f"Analyzing sentiment for {text_col}")

                sentiments = []
                compounds = []

                for text in df[text_col]:
                    if pd.isna(text) or text == "":
                        sentiments.append(0.0)
                        compounds.append(0.0)
                    else:
                        try:
                            scores = self.vader_analyzer.polarity_scores(str(text))
                            # Use compound score as primary sentiment
                            compounds.append(scores["compound"])
                            # Calculate weighted sentiment
                            sentiment = scores["pos"] - scores["neg"]
                            sentiments.append(sentiment)
                        except Exception as e:
                            self.logger.warning(f"Error analyzing sentiment: {str(e)}")
                            sentiments.append(0.0)
                            compounds.append(0.0)

                df[sentiment_col] = sentiments
                df[compound_col] = compounds

        # Add categorical sentiment labels
        df["sentiment_label"] = df["combined_sentiment_compound"].apply(
            self._categorize_sentiment
        )

        # Add sentiment features
        df = self._add_sentiment_features(df)

        self.logger.info(f"Completed sentiment analysis for {len(df)} news articles")
        return df

    def analyze_sec_sentiment(self, sec_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment of SEC filings using FinBERT

        Args:
            sec_df: DataFrame with SEC filings data

        Returns:
            DataFrame with sentiment scores added
        """
        self.logger.info("Analyzing SEC filings sentiment with FinBERT...")

        if sec_df.empty:
            return sec_df

        df = sec_df.copy()

        # Initialize sentiment columns
        df["mda_sentiment"] = 0.0
        df["mda_sentiment_label"] = "neutral"
        df["risk_factors_sentiment"] = 0.0
        df["risk_factors_sentiment_label"] = "neutral"

        # Analyze MD&A sentiment
        if "mda_text" in df.columns and self.finbert_model is not None:
            self.logger.info("Analyzing MD&A sentiment")
            mda_sentiments, mda_labels = self._analyze_finbert_sentiment(df["mda_text"])
            df["mda_sentiment"] = mda_sentiments
            df["mda_sentiment_label"] = mda_labels
        elif "mda_text" in df.columns and self.vader_analyzer is not None:
            # Fallback to VADER if FinBERT not available
            self.logger.info("Using VADER for MD&A sentiment (FinBERT not available)")
            df = self._analyze_vader_fallback(
                df, "mda_text", "mda_sentiment", "mda_sentiment_label"
            )

        # Analyze Risk Factors sentiment
        if "risk_factors_text" in df.columns and self.finbert_model is not None:
            self.logger.info("Analyzing Risk Factors sentiment")
            risk_sentiments, risk_labels = self._analyze_finbert_sentiment(
                df["risk_factors_text"]
            )
            df["risk_factors_sentiment"] = risk_sentiments
            df["risk_factors_sentiment_label"] = risk_labels
        elif "risk_factors_text" in df.columns and self.vader_analyzer is not None:
            # Fallback to VADER if FinBERT not available
            self.logger.info(
                "Using VADER for Risk Factors sentiment (FinBERT not available)"
            )
            df = self._analyze_vader_fallback(
                df,
                "risk_factors_text",
                "risk_factors_sentiment",
                "risk_factors_sentiment_label",
            )

        # Add combined sentiment features
        df = self._add_sec_sentiment_features(df)

        self.logger.info(f"Completed SEC sentiment analysis for {len(df)} filings")
        return df

    def _analyze_finbert_sentiment(
        self, texts: pd.Series
    ) -> Tuple[List[float], List[str]]:
        """Analyze sentiment using FinBERT model"""
        sentiments = []
        labels = []

        try:
            import torch

            for text in texts:
                if pd.isna(text) or text == "":
                    sentiments.append(0.0)
                    labels.append("neutral")
                    continue

                try:
                    # Truncate text if too long (FinBERT has token limits)
                    text = str(text)[:512]  # Approximate token limit

                    # Tokenize
                    if self.finbert_tokenizer is None:
                        return 0.0
                    inputs = self.finbert_tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=512,
                    )

                    # Get prediction
                    if self.finbert_model is None:
                        return 0.0
                    with torch.no_grad():
                        outputs = self.finbert_model(**inputs)
                        predictions = torch.nn.functional.softmax(
                            outputs.logits, dim=-1
                        )

                    # FinBERT returns: [negative, neutral, positive]
                    probs = predictions.cpu().numpy()[0]

                    # Calculate sentiment score (-1 to 1)
                    sentiment_score = probs[2] - probs[0]  # positive - negative
                    sentiments.append(float(sentiment_score))

                    # Determine label
                    max_idx = np.argmax(probs)
                    if max_idx == 0:
                        labels.append("negative")
                    elif max_idx == 1:
                        labels.append("neutral")
                    else:
                        labels.append("positive")

                except Exception as e:
                    self.logger.warning(f"Error in FinBERT analysis: {str(e)}")
                    sentiments.append(0.0)
                    labels.append("neutral")

        except Exception as e:
            self.logger.error(f"FinBERT analysis failed: {str(e)}")
            sentiments = [0.0] * len(texts)
            labels = ["neutral"] * len(texts)

        return sentiments, labels

    def _analyze_vader_fallback(
        self, df: pd.DataFrame, text_col: str, sentiment_col: str, label_col: str
    ) -> pd.DataFrame:
        """Fallback sentiment analysis using VADER"""
        if not self.vader_analyzer:
            df[sentiment_col] = 0.0
            df[label_col] = "neutral"
            return df

        sentiments = []
        labels = []

        for text in df[text_col]:
            if pd.isna(text) or text == "":
                sentiments.append(0.0)
                labels.append("neutral")
            else:
                try:
                    scores = self.vader_analyzer.polarity_scores(str(text))
                    sentiment = scores["compound"]
                    sentiments.append(sentiment)
                    labels.append(self._categorize_sentiment(sentiment))
                except Exception as e:
                    sentiments.append(0.0)
                    labels.append("neutral")

        df[sentiment_col] = sentiments
        df[label_col] = labels

        return df

    def _categorize_sentiment(self, compound_score: float) -> str:
        """Categorize sentiment based on compound score"""
        if compound_score >= 0.05:
            return "positive"
        elif compound_score <= -0.05:
            return "negative"
        else:
            return "neutral"

    def _add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add additional sentiment-based features"""

        # Sentiment intensity (absolute value)
        if "combined_sentiment_compound" in df.columns:
            df["sentiment_intensity"] = abs(df["combined_sentiment_compound"])

        # Sentiment consistency across title and description
        if all(
            col in df.columns
            for col in ["title_sentiment_compound", "description_sentiment_compound"]
        ):
            df["sentiment_consistency"] = 1 - abs(
                df["title_sentiment_compound"] - df["description_sentiment_compound"]
            )

        # Daily sentiment aggregations (if date available)
        if all(
            col in df.columns
            for col in ["published_date", "symbol", "combined_sentiment_compound"]
        ):
            df["date"] = df["published_date"].dt.date

            # Daily average sentiment by symbol
            daily_sentiment = (
                df.groupby(["symbol", "date"])["combined_sentiment_compound"]
                .mean()
                .reset_index()
            )
            daily_sentiment.columns = ["symbol", "date", "daily_avg_sentiment"]

            df = df.merge(daily_sentiment, on=["symbol", "date"], how="left")

            # Sentiment volatility (rolling standard deviation)
            df = df.sort_values(["symbol", "published_date"])
            df["sentiment_volatility"] = (
                df.groupby("symbol")["combined_sentiment_compound"]
                .rolling(window=7, min_periods=1)
                .std()
                .reset_index(0, drop=True)
            )

        return df

    def _add_sec_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add SEC-specific sentiment features"""

        # Combined sentiment score
        sentiment_cols = ["mda_sentiment", "risk_factors_sentiment"]
        existing_cols = [col for col in sentiment_cols if col in df.columns]

        if existing_cols:
            df["combined_sec_sentiment"] = df[existing_cols].mean(axis=1)
            df["combined_sec_sentiment_label"] = df["combined_sec_sentiment"].apply(
                self._categorize_sentiment
            )

        # Risk sentiment change over time
        if all(
            col in df.columns
            for col in ["symbol", "filing_date", "risk_factors_sentiment"]
        ):
            df = df.sort_values(["symbol", "filing_date"])
            df["risk_sentiment_change"] = df.groupby("symbol")[
                "risk_factors_sentiment"
            ].diff()

        # Sentiment intensity for each section
        for col in ["mda_sentiment", "risk_factors_sentiment"]:
            if col in df.columns:
                df[f"{col}_intensity"] = abs(df[col])

        return df

    def aggregate_sentiment_by_date(
        self, news_df: pd.DataFrame, sec_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Aggregate sentiment scores by date and symbol

        Args:
            news_df: News DataFrame with sentiment scores
            sec_df: SEC DataFrame with sentiment scores (optional)

        Returns:
            DataFrame with daily sentiment aggregations
        """
        self.logger.info("Aggregating sentiment by date...")

        aggregations = []

        # News sentiment aggregation
        if (
            news_df is not None
            and not news_df.empty
            and "published_date" in news_df.columns
        ):
            news_df["date"] = news_df["published_date"].dt.date

            news_agg = (
                news_df.groupby(["symbol", "date"])
                .agg(
                    {
                        "combined_sentiment_compound": ["mean", "std", "count"],
                        "sentiment_intensity": "mean",
                        "title_sentiment_compound": "mean",
                        "description_sentiment_compound": "mean",
                    }
                )
                .reset_index()
            )

            # Flatten column names
            news_agg.columns = [
                "symbol",
                "date",
                "news_sentiment_mean",
                "news_sentiment_std",
                "news_count",
                "news_sentiment_intensity",
                "news_title_sentiment",
                "news_description_sentiment",
            ]

            news_agg["data_source"] = "news"
            aggregations.append(news_agg)

        # SEC sentiment aggregation
        if sec_df is not None and not sec_df.empty and "filing_date" in sec_df.columns:
            sec_df["date"] = sec_df["filing_date"].dt.date

            sec_agg = (
                sec_df.groupby(["symbol", "date"])
                .agg(
                    {
                        "mda_sentiment": "mean",
                        "risk_factors_sentiment": "mean",
                        "combined_sec_sentiment": "mean",
                    }
                )
                .reset_index()
            )

            sec_agg.columns = [
                "symbol",
                "date",
                "sec_mda_sentiment",
                "sec_risk_sentiment",
                "sec_combined_sentiment",
            ]

            sec_agg["data_source"] = "sec"
            aggregations.append(sec_agg)

        # Combine all aggregations
        if aggregations:
            if len(aggregations) == 1:
                result_df = aggregations[0]
            else:
                # Merge news and SEC data
                result_df = aggregations[0].merge(
                    aggregations[1],
                    on=["symbol", "date"],
                    how="outer",
                    suffixes=("_news", "_sec"),
                )
        else:
            result_df = pd.DataFrame()

        self.logger.info(
            f"Created sentiment aggregations for {len(result_df)} symbol-date combinations"
        )
        return result_df

    def calculate_sentiment_indicators(
        self, sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate sentiment-based risk indicators

        Args:
            sentiment_df: DataFrame with sentiment data

        Returns:
            DataFrame with sentiment indicators
        """
        if sentiment_df.empty:
            return sentiment_df

        df = sentiment_df.copy()

        # Convert date to datetime for calculations
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values(["symbol", "date"])

        # Sentiment moving averages
        sentiment_cols = [
            col for col in df.columns if "sentiment" in col and col.endswith("_mean")
        ]

        for col in sentiment_cols:
            if col in df.columns:
                base_name = col.replace("_mean", "")

                # 7-day and 30-day moving averages
                df[f"{base_name}_ma7"] = (
                    df.groupby("symbol")[col]
                    .rolling(window=7, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )

                df[f"{base_name}_ma30"] = (
                    df.groupby("symbol")[col]
                    .rolling(window=30, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )

                # Sentiment trend (difference between short and long MA)
                df[f"{base_name}_trend"] = (
                    df[f"{base_name}_ma7"] - df[f"{base_name}_ma30"]
                )

        # Sentiment volatility indicators
        if "news_sentiment_std" in df.columns:
            df["sentiment_volatility_ma7"] = (
                df.groupby("symbol")["news_sentiment_std"]
                .rolling(window=7, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )

        # News volume indicators
        if "news_count" in df.columns:
            df["news_volume_ma7"] = (
                df.groupby("symbol")["news_count"]
                .rolling(window=7, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )

            # High news volume periods (potential risk indicator)
            df["high_news_volume"] = (
                df["news_count"]
                > df.groupby("symbol")["news_count"].transform("mean") * 2
            ).astype(int)

        return df

    def save_sentiment_data(
        self,
        news_df: pd.DataFrame,
        sec_df: pd.DataFrame,
        aggregated_df: pd.DataFrame,
        output_dir: str,
    ):
        """Save sentiment analysis results"""
        os.makedirs(output_dir, exist_ok=True)

        # Save news sentiment
        if not news_df.empty:
            news_sentiment_path = os.path.join(output_dir, "news_sentiment.csv")
            news_df.to_csv(news_sentiment_path, index=False)

        # Save SEC sentiment
        if not sec_df.empty:
            sec_sentiment_path = os.path.join(output_dir, "sec_sentiment.csv")
            sec_df.to_csv(sec_sentiment_path, index=False)

        # Save aggregated sentiment
        if not aggregated_df.empty:
            agg_sentiment_path = os.path.join(output_dir, "sentiment_aggregated.csv")
            aggregated_df.to_csv(agg_sentiment_path, index=False)

        self.logger.info(f"Saved sentiment analysis results to {output_dir}")
