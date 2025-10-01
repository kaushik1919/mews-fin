"""
News data collector module
Fetches news headlines from GNews API and other free sources
"""

import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from src.utils.logging import get_logger

try:
    from src.research.robustness import SentimentBiasDetector
except ImportError:  # pragma: no cover - optional research dependency
    SentimentBiasDetector = None  # type: ignore


class NewsDataCollector:
    """Collects news data from various free APIs"""

    def __init__(
        self, gnews_api_key: Optional[str] = None, news_api_key: Optional[str] = None
    ):
        self.gnews_api_key = gnews_api_key
        self.news_api_key = news_api_key
        self.logger = get_logger(__name__)
        self.rate_limit_delay = 6  # seconds between requests (10 per minute for GNews)
        self.bias_detector = (
            SentimentBiasDetector(sentiment_col="sentiment_score")
            if SentimentBiasDetector is not None
            else None
        )

        # Base URLs
        self.gnews_base_url = "https://gnews.io/api/v4"
        self.newsapi_base_url = "https://newsapi.org/v2"

    def fetch_gnews_data(
        self,
        symbol: str,
        company_name: str,
        start_date: str,
        end_date: str,
        max_articles: int = 100,
    ) -> List[Dict]:
        """
        Fetch news from GNews API

        Args:
            symbol: Stock symbol
            company_name: Company name for search
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            max_articles: Maximum number of articles to fetch

        Returns:
            List of news articles
        """
        if not self.gnews_api_key:
            self.logger.warning("GNews API key not provided")
            return []

        articles = []
        raw_responses = []

        try:
            # Search queries for the company
            queries = [
                f"{company_name}",
                f"{symbol}",
                f"{company_name} stock",
                f"{company_name} earnings",
                f"{company_name} financial",
            ]

            for query in queries:
                self.logger.info(f"Fetching GNews for query: {query}")

                # GNews API parameters
                params = {
                    "q": query,
                    "token": self.gnews_api_key,
                    "lang": "en",
                    "country": "us",
                    "max": min(
                        max_articles // len(queries), 10
                    ),  # GNews free tier limit
                    "from": start_date,
                    "to": end_date,
                    "sortby": "publishedAt",
                }

                # Make request
                response = requests.get(f"{self.gnews_base_url}/search", params=params)  # type: ignore

                if response.status_code == 200:
                    data = response.json()
                    raw_responses.append(data)

                    if "articles" in data:
                        for article in data["articles"]:
                            articles.append(
                                {
                                    "symbol": symbol,
                                    "company_name": company_name,
                                    "title": article.get("title", ""),
                                    "description": article.get("description", ""),
                                    "content": article.get("content", ""),
                                    "url": article.get("url", ""),
                                    "published_date": article.get("publishedAt", ""),
                                    "source": article.get("source", {}).get("name", ""),
                                    "query_used": query,
                                    "api_source": "gnews",
                                }
                            )
                else:
                    self.logger.warning(
                        f"GNews API error: {response.status_code} - {response.text}"
                    )

                # Rate limiting
                time.sleep(self.rate_limit_delay)

        except Exception as e:
            self.logger.error(f"Error fetching GNews data for {symbol}: {str(e)}")

        return articles

    def fetch_newsapi_data(
        self,
        symbol: str,
        company_name: str,
        start_date: str,
        end_date: str,
        max_articles: int = 100,
    ) -> List[Dict]:
        """
        Fetch news from News API (alternative source)

        Args:
            symbol: Stock symbol
            company_name: Company name for search
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            max_articles: Maximum number of articles to fetch

        Returns:
            List of news articles
        """
        if not self.news_api_key:
            self.logger.warning("News API key not provided")
            return []

        articles = []
        raw_responses = []

        try:
            # Search query
            query = f"{company_name} OR {symbol}"

            params = {
                "q": query,
                "apiKey": self.news_api_key,
                "language": "en",
                "sortBy": "publishedAt",
                "from": start_date,
                "to": end_date,
                "pageSize": min(max_articles, 100),
            }

            response = requests.get(
                f"{self.newsapi_base_url}/everything", params=params  # type: ignore
            )

            if response.status_code == 200:
                data = response.json()
                raw_responses.append(data)

                if "articles" in data:
                    for article in data["articles"]:
                        articles.append(
                            {
                                "symbol": symbol,
                                "company_name": company_name,
                                "title": article.get("title", ""),
                                "description": article.get("description", ""),
                                "content": article.get("content", ""),
                                "url": article.get("url", ""),
                                "published_date": article.get("publishedAt", ""),
                                "source": article.get("source", {}).get("name", ""),
                                "query_used": query,
                                "api_source": "newsapi",
                            }
                        )
            else:
                self.logger.warning(
                    f"News API error: {response.status_code} - {response.text}"
                )

        except Exception as e:
            self.logger.error(f"Error fetching News API data for {symbol}: {str(e)}")

        return articles

    def fetch_yahoo_news(self, symbol: str, company_name: str) -> List[Dict]:
        """
        Fetch news from Yahoo Finance (free, no API key required)

        Args:
            symbol: Stock symbol
            company_name: Company name

        Returns:
            List of news articles
        """
        articles = []

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            news = ticker.news

            for article in news:
                articles.append(
                    {
                        "symbol": symbol,
                        "company_name": company_name,
                        "title": article.get("title", ""),
                        "description": article.get("summary", ""),
                        "content": "",  # Yahoo doesn't provide full content
                        "url": article.get("link", ""),
                        "published_date": (
                            datetime.fromtimestamp(
                                article.get("providerPublishTime", 0)
                            ).isoformat()
                            if article.get("providerPublishTime")
                            else ""
                        ),
                        "source": article.get("publisher", ""),
                        "query_used": symbol,
                        "api_source": "yahoo",
                    }
                )

        except Exception as e:
            self.logger.error(f"Error fetching Yahoo news for {symbol}: {str(e)}")

        return articles

    def fetch_reddit_mentions(
        self, symbol: str, company_name: str, subreddits: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Fetch Reddit mentions (using free Reddit API)

        Args:
            symbol: Stock symbol
            company_name: Company name
            subreddits: List of subreddits to search

        Returns:
            List of Reddit posts/comments
        """
        if subreddits is None:
            subreddits = [
                "investing",
                "stocks",
                "SecurityAnalysis",
                "financialindependence",
            ]

        mentions = []

        try:
            for subreddit in subreddits:
                # Use Reddit's JSON API (no authentication required for public posts)
                url = f"https://www.reddit.com/r/{subreddit}/search.json"
                params = {
                    "q": f"{symbol} OR {company_name}",
                    "sort": "new",
                    "limit": 25,
                    "restrict_sr": 1,
                }

                headers = {"User-Agent": "Market Risk System 1.0"}
                response = requests.get(url, params=params, headers=headers)  # type: ignore

                if response.status_code == 200:
                    data = response.json()

                    for post in data.get("data", {}).get("children", []):
                        post_data = post.get("data", {})

                        mentions.append(
                            {
                                "symbol": symbol,
                                "company_name": company_name,
                                "title": post_data.get("title", ""),
                                "description": post_data.get("selftext", ""),
                                "content": "",
                                "url": f"https://reddit.com{post_data.get('permalink', '')}",
                                "published_date": (
                                    datetime.fromtimestamp(
                                        post_data.get("created_utc", 0)
                                    ).isoformat()
                                    if post_data.get("created_utc")
                                    else ""
                                ),
                                "source": f"r/{subreddit}",
                                "query_used": f"{symbol} OR {company_name}",
                                "api_source": "reddit",
                                "score": post_data.get("score", 0),
                                "num_comments": post_data.get("num_comments", 0),
                            }
                        )

                # Rate limiting for Reddit
                time.sleep(2)

        except Exception as e:
            self.logger.error(f"Error fetching Reddit mentions for {symbol}: {str(e)}")

        return mentions

    def get_company_name_mapping(self, symbols: List[str]) -> Dict[str, str]:
        """
        Get company names for stock symbols

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary mapping symbol to company name
        """
        mapping = {}

        try:
            import yfinance as yf

            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    company_name = info.get("longName", info.get("shortName", symbol))
                    mapping[symbol] = company_name

                    time.sleep(0.1)  # Rate limiting

                except Exception as e:
                    self.logger.warning(
                        f"Could not get company name for {symbol}: {str(e)}"
                    )
                    mapping[symbol] = symbol

        except Exception as e:
            self.logger.error(f"Error creating company name mapping: {str(e)}")
            # Fallback: use symbols as names
            mapping = {symbol: symbol for symbol in symbols}

        return mapping

    def fetch_news_for_symbols(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        max_articles_per_symbol: int = 50,
        use_all_sources: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch news for multiple symbols from all available sources

        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            max_articles_per_symbol: Maximum articles per symbol
            use_all_sources: Whether to use all available news sources

        Returns:
            DataFrame with all news data
        """
        all_news = []

        # Get company name mapping
        self.logger.info("Getting company names for symbols...")
        company_mapping = self.get_company_name_mapping(symbols)

        for symbol in symbols:
            company_name = company_mapping.get(symbol, symbol)
            self.logger.info(f"Fetching news for {symbol} ({company_name})")

            try:
                # Fetch from GNews if available
                if use_all_sources and self.gnews_api_key:
                    gnews_articles = self.fetch_gnews_data(
                        symbol,
                        company_name,
                        start_date,
                        end_date,
                        max_articles_per_symbol // 3,
                    )
                    all_news.extend(gnews_articles)

                # Fetch from News API if available
                if use_all_sources and self.news_api_key:
                    newsapi_articles = self.fetch_newsapi_data(
                        symbol,
                        company_name,
                        start_date,
                        end_date,
                        max_articles_per_symbol // 3,
                    )
                    all_news.extend(newsapi_articles)

                # Always try Yahoo Finance (free)
                yahoo_articles = self.fetch_yahoo_news(symbol, company_name)
                all_news.extend(yahoo_articles)

                # Fetch Reddit mentions if enabled
                if use_all_sources:
                    reddit_mentions = self.fetch_reddit_mentions(symbol, company_name)
                    all_news.extend(reddit_mentions)

            except Exception as e:
                self.logger.error(f"Error fetching news for {symbol}: {str(e)}")
                continue

        # Create DataFrame
        news_df = pd.DataFrame(all_news)

        if not news_df.empty:
            # Clean and standardize data
            news_df = self._clean_news_data(news_df)

            # Remove duplicates
            news_df = news_df.drop_duplicates(subset=["title", "url"], keep="first")

            self.logger.info(
                f"Collected {len(news_df)} news articles for {len(symbols)} symbols"
            )

        return news_df

    def _clean_news_data(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize news data"""
        if news_df.empty:
            return news_df

        # Clean text fields
        text_columns = ["title", "description", "content"]
        for col in text_columns:
            if col in news_df.columns:
                news_df[col] = news_df[col].fillna("")
                news_df[col] = news_df[col].str.replace(r"\s+", " ", regex=True)
                news_df[col] = news_df[col].str.strip()

        # Standardize dates
        if "published_date" in news_df.columns:
            news_df["published_date"] = pd.to_datetime(
                news_df["published_date"], errors="coerce"
            )

        # Add text length metrics
        news_df["title_length"] = news_df["title"].str.len()
        news_df["description_length"] = news_df["description"].str.len()

        # Create combined text for analysis
        news_df["combined_text"] = (
            news_df["title"] + " " + news_df["description"] + " " + news_df["content"]
        ).str.strip()

        return news_df

    def save_news_data(self, news_df: pd.DataFrame, output_dir: str):
        """
        Save news data to files

        Args:
            news_df: DataFrame with news data
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save main news data
        news_path = os.path.join(output_dir, "news_data.csv")
        news_df.to_csv(news_path, index=False)

        # Save by source for analysis
        if not news_df.empty and "api_source" in news_df.columns:
            for source in news_df["api_source"].unique():
                source_df = news_df[news_df["api_source"] == source]
                source_path = os.path.join(output_dir, f"news_data_{source}.csv")
                source_df.to_csv(source_path, index=False)

        self.logger.info(f"Saved {len(news_df)} news articles to {output_dir}")

    def get_news_statistics(self, news_df: pd.DataFrame) -> Dict:
        """
        Get statistics about collected news data

        Args:
            news_df: DataFrame with news data

        Returns:
            Dictionary with statistics
        """
        if news_df.empty:
            return {}

        stats = {
            "total_articles": len(news_df),
            "unique_companies": (
                news_df["symbol"].nunique() if "symbol" in news_df.columns else 0
            ),
            "sources": (
                news_df["api_source"].value_counts().to_dict()
                if "api_source" in news_df.columns
                else {}
            ),
            "date_range": {
                "earliest": (
                    news_df["published_date"].min()
                    if "published_date" in news_df.columns
                    else None
                ),
                "latest": (
                    news_df["published_date"].max()
                    if "published_date" in news_df.columns
                    else None
                ),
            },
            "avg_title_length": (
                news_df["title_length"].mean()
                if "title_length" in news_df.columns
                else 0
            ),
            "avg_description_length": (
                news_df["description_length"].mean()
                if "description_length" in news_df.columns
                else 0
            ),
        }

        return stats
