"""Utility script to pull fresh news sentiment using the configured GNews API key.

The script fetches recent articles for a handful of showcase tickers, applies the
project's sentiment pipeline, and writes a compact snapshot that can be used to
refresh README examples or dashboards.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.news_collector import NewsDataCollector
from src.sentiment_analyzer import SentimentAnalyzer


@dataclass
class SnapshotConfig:
    symbols: List[str]
    lookback_days: int = 7
    max_articles_per_symbol: int = 45
    output_path: str = "outputs/news_sentiment_snapshot.csv"
    aggregated_output_path: str = "outputs/news_sentiment_timeseries.csv"
    plot_path: str = "outputs/sentiment_analysis.png"


def _extract_latest_rows(
    df: pd.DataFrame, group_key: str, date_col: str
) -> pd.DataFrame:
    """Return the most recent row per group based on *date_col*."""

    if df.empty:
        return df

    ordered = df.sort_values(by=[group_key, date_col], ascending=[True, False])
    return ordered.groupby(group_key, as_index=False).first()


def _heuristic_risk_probability(sentiment: pd.Series) -> pd.Series:
    """Compute a smooth sigmoid-based risk probability from sentiment values."""
    import numpy as np

    # Clip sentiment to avoid blowing up exponentials and flip sign so negative -> higher risk
    clipped = sentiment.clip(lower=-1.0, upper=1.0)
    return 1 / (1 + np.exp(3.5 * clipped))


def _load_latest_sec_sentiment(symbols: Iterable[str]) -> pd.DataFrame:
    """Load the latest SEC sentiment entries for the provided symbols."""

    sec_path = os.path.join("data", "sec_sentiment.csv")
    if not os.path.exists(sec_path):
        return pd.DataFrame()

    sec_df = pd.read_csv(sec_path, parse_dates=["filing_date", "date"])
    sec_df = sec_df[sec_df["symbol"].isin(symbols)].copy()
    if sec_df.empty:
        return sec_df

    latest_sec = _extract_latest_rows(sec_df, "symbol", "filing_date")
    return latest_sec[
        [
            "symbol",
            "filing_date",
            "mda_sentiment",
            "risk_factors_sentiment",
            "combined_sec_sentiment",
        ]
    ]


def collect_news_sentiment(
    config: SnapshotConfig, include_aggregated: bool = False
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and score news sentiment for the configured symbols."""

    end_date = date.today()
    start_date = end_date - timedelta(days=config.lookback_days)

    collector = NewsDataCollector(gnews_api_key=os.environ.get("GNEWS_API_KEY"))

    news_df = collector.fetch_news_for_symbols(
        symbols=config.symbols,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        max_articles_per_symbol=config.max_articles_per_symbol,
        use_all_sources=True,
    )

    if news_df.empty:
        return news_df

    analyzer = SentimentAnalyzer()
    scored = analyzer.analyze_news_sentiment(news_df)

    sentiment_daily = analyzer.aggregate_sentiment_by_date(scored)
    if sentiment_daily.empty:
        return (
            (pd.DataFrame(), pd.DataFrame()) if include_aggregated else sentiment_daily
        )

    renamed = sentiment_daily.rename(
        columns={
            "symbol": "Symbol",
            "date": "Date",
            "news_sentiment_mean": "News_Sentiment",
            "news_sentiment_std": "News_Sentiment_Volatility",
            "news_count": "Article_Count",
        }
    )

    renamed["Date"] = pd.to_datetime(renamed["Date"])

    # Attach a representative headline for context (latest by published date)
    if "published_date" in scored.columns:
        scored["published_date"] = pd.to_datetime(
            scored["published_date"], errors="coerce"
        )
        scored_sorted = scored.sort_values(
            ["symbol", "published_date"], ascending=[True, False]
        )
        headline_df = scored_sorted.groupby("symbol", as_index=False).first()[
            ["symbol", "title", "url"]
        ]
        headline_df = headline_df.rename(
            columns={
                "symbol": "Symbol",
                "title": "Headline",
                "url": "Article_URL",
            }
        )
        renamed = renamed.merge(headline_df, on="Symbol", how="left")

    # Merge SEC sentiment if available
    sec_sentiment = _load_latest_sec_sentiment(config.symbols)
    if not sec_sentiment.empty:
        sec_sentiment = sec_sentiment.rename(columns={"symbol": "Symbol"})
        renamed = renamed.merge(sec_sentiment, on="Symbol", how="left")

    # Combined sentiment: average of SEC combined sentiment (when present) and the news mean
    if "combined_sec_sentiment" in renamed.columns:
        renamed["Combined_Sentiment"] = renamed[
            ["combined_sec_sentiment", "News_Sentiment"]
        ].mean(axis=1)
        renamed["Combined_Sentiment"] = renamed["Combined_Sentiment"].fillna(
            renamed["News_Sentiment"]
        )
    else:
        renamed["Combined_Sentiment"] = renamed["News_Sentiment"]

    # Estimate risk probability using a sigmoid heuristic so that negative sentiment maps to higher risk
    renamed["Risk_Probability"] = _heuristic_risk_probability(
        renamed["Combined_Sentiment"]
    )

    # Round for readability
    for col in [
        "News_Sentiment",
        "News_Sentiment_Volatility",
        "mda_sentiment",
        "risk_factors_sentiment",
        "combined_sec_sentiment",
        "Combined_Sentiment",
        "Risk_Probability",
    ]:
        if col in renamed.columns:
            renamed[col] = renamed[col].round(3)

    aggregated = renamed.copy()

    resampled_groups = []
    for symbol, group in aggregated.groupby("Symbol"):
        group = group.sort_values("Date").set_index("Date")
        date_range = pd.date_range(group.index.min(), group.index.max(), freq="D")
        reindexed = group.reindex(date_range)
        reindexed["Symbol"] = symbol
        reindexed = reindexed.fillna(method="ffill").fillna(method="bfill")
        resampled_groups.append(
            reindexed.reset_index().rename(columns={"index": "Date"})
        )

    aggregated_resampled = (
        pd.concat(resampled_groups, ignore_index=True)
        if resampled_groups
        else aggregated
    )

    # Recalculate risk probability after resample to ensure numeric stability
    if "Combined_Sentiment" in aggregated_resampled.columns:
        aggregated_resampled["Risk_Probability"] = _heuristic_risk_probability(
            aggregated_resampled["Combined_Sentiment"]
        )

    # Round numeric columns for readability
    numeric_cols = {
        "News_Sentiment",
        "News_Sentiment_Volatility",
        "mda_sentiment",
        "risk_factors_sentiment",
        "combined_sec_sentiment",
        "Combined_Sentiment",
        "Risk_Probability",
    }

    for col in numeric_cols:
        if col in aggregated_resampled.columns:
            aggregated_resampled[col] = aggregated_resampled[col].round(3)

    latest = _extract_latest_rows(aggregated_resampled.copy(), "Symbol", "Date")
    latest["Date"] = latest["Date"].dt.date

    if include_aggregated:
        return latest, aggregated_resampled

    return latest


def create_sentiment_timeline_plot(
    aggregated: pd.DataFrame, output_path: str
) -> Optional[str]:
    """Generate a dual-axis sentiment vs risk timeline chart."""

    if aggregated.empty:
        return None

    try:
        import matplotlib.pyplot as plt

        working = aggregated.copy()
        working["Date"] = pd.to_datetime(working["Date"])

        if "Combined_Sentiment" in working.columns:
            daily = working.groupby("Date")["Combined_Sentiment"].mean().reset_index()
        else:
            daily = working.groupby("Date")["News_Sentiment"].mean().reset_index()
            daily = daily.rename(columns={daily.columns[1]: "Combined_Sentiment"})

        if daily.empty:
            return None

        daily = daily.sort_values("Date")
        daily["Risk_Probability"] = _heuristic_risk_probability(
            daily["Combined_Sentiment"]
        )

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()

        ax1.plot(
            daily["Date"],
            daily["Combined_Sentiment"],
            color="#2563eb",
            linewidth=2,
            label="Sentiment",
        )
        ax1.axhline(0, color="#94a3b8", linestyle="--", linewidth=1)
        ax1.set_ylabel("Average Sentiment", color="#2563eb")
        ax1.tick_params(axis="y", labelcolor="#2563eb")

        ax2.plot(
            daily["Date"],
            daily["Risk_Probability"],
            color="#dc2626",
            linewidth=2,
            label="Risk Probability",
        )
        ax2.set_ylabel("Risk Probability", color="#dc2626")
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis="y", labelcolor="#dc2626")

        ax1.set_title("Sentiment vs Risk Probability Timeline")
        ax1.set_xlabel("Date")
        fig.autofmt_xdate()

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper left")

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path
    except Exception:
        return None


def main(config: Optional[SnapshotConfig] = None) -> None:
    load_dotenv()

    gnews_key = os.environ.get("GNEWS_API_KEY")
    if not gnews_key:
        raise SystemExit(
            "GNEWS_API_KEY missing. Copy config/.env.template to .env and provide your key."
        )

    if config is None:
        parsed_args = _parse_args()
        config = SnapshotConfig(
            symbols=parsed_args.symbols,
            lookback_days=parsed_args.lookback,
            max_articles_per_symbol=parsed_args.max_articles,
            output_path=parsed_args.output,
            aggregated_output_path=parsed_args.aggregated_output,
            plot_path=parsed_args.plot_output,
        )

    result = collect_news_sentiment(config, include_aggregated=True)

    if isinstance(result, tuple):
        snapshot_df, aggregated_df = result
    else:
        snapshot_df = result
        aggregated_df = pd.DataFrame()

    if snapshot_df.empty:
        raise SystemExit(
            "No news articles were returned by GNews. Try increasing the lookback window or verifying the API quota."
        )

    output_dir = os.path.dirname(config.output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    snapshot_df.to_csv(config.output_path, index=False)

    if not aggregated_df.empty:
        aggregated_df_sorted = aggregated_df.sort_values(["Date", "Symbol"])
        aggregated_df_sorted.to_csv(config.aggregated_output_path, index=False)
        plot_path = create_sentiment_timeline_plot(
            aggregated_df_sorted, config.plot_path
        )
        if plot_path:
            print(f"Saved sentiment timeline to {plot_path}")

    print(f"Saved {len(snapshot_df)} sentiment rows to {config.output_path}")
    print(snapshot_df)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a news sentiment snapshot using GNews"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["MSFT", "GOOGL", "NVDA"],
        help="Ticker symbols to query",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=10,
        help="Lookback window in days",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=45,
        help="Maximum number of articles to request per symbol across all GNews queries",
    )
    parser.add_argument(
        "--output",
        default="outputs/news_sentiment_snapshot.csv",
        help="Where to write the snapshot CSV",
    )
    parser.add_argument(
        "--aggregated-output",
        default="outputs/news_sentiment_timeseries.csv",
        help="Optional path for saving the aggregated sentiment time series",
    )
    parser.add_argument(
        "--plot-output",
        default="outputs/sentiment_analysis.png",
        help="File path for the generated sentiment timeline plot",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
