"""
Backtesting system for market risk prediction
Validates system performance against historical market events (COVID crash, etc.)
"""

import json
import logging
import os
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class RiskBacktester:
    """Backtests the early warning system against historical market events"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.backtest_results = {}

        # Define major market events for validation
        self.market_events = {
            "covid_crash": {
                "start_date": "2020-02-20",
                "end_date": "2020-03-23",
                "description": "COVID-19 Market Crash",
                "type": "crash",
            },
            "covid_recovery": {
                "start_date": "2020-03-23",
                "end_date": "2020-06-01",
                "description": "COVID Recovery Period",
                "type": "recovery",
            },
            "fed_tightening_2022": {
                "start_date": "2022-01-01",
                "end_date": "2022-06-30",
                "description": "Fed Tightening Period 2022",
                "type": "volatility",
            },
            "banking_crisis_2023": {
                "start_date": "2023-03-01",
                "end_date": "2023-04-30",
                "description": "Banking Sector Crisis 2023",
                "type": "sector_crisis",
            },
        }

    def run_comprehensive_backtest(
        self,
        df: pd.DataFrame,
        predictions_df: Optional[pd.DataFrame] = None,
        start_date: str = "2019-01-01",
        end_date: str = "2023-12-31",
    ) -> Dict[str, Any]:
        """
        Run comprehensive backtesting analysis

        Args:
            df: Integrated dataset with features and labels
            predictions_df: DataFrame with model predictions
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            Dictionary with backtest results
        """
        self.logger.info(
            f"Running comprehensive backtest from {start_date} to {end_date}"
        )

        if df.empty:
            self.logger.error("No data provided for backtesting")
            return {}

        # Filter data for backtest period
        backtest_data = self._prepare_backtest_data(df, start_date, end_date)

        if backtest_data.empty:
            self.logger.error("No data in backtest period")
            return {}

        results = {
            "backtest_period": {"start": start_date, "end": end_date},
            "data_summary": self._get_data_summary(backtest_data),
            "event_analysis": {},
            "prediction_accuracy": {},
            "early_warning_performance": {},
            "portfolio_simulation": {},
            "risk_metrics": {},
        }

        # 1. Analyze performance during major market events
        results["event_analysis"] = self._analyze_market_events(
            backtest_data, predictions_df
        )

        # 2. Calculate prediction accuracy metrics
        if predictions_df is not None:
            results["prediction_accuracy"] = self._calculate_prediction_accuracy(
                backtest_data, predictions_df
            )

        # 3. Evaluate early warning capabilities
        results["early_warning_performance"] = self._evaluate_early_warning(
            backtest_data
        )

        # 4. Simulate portfolio performance using risk signals
        results["portfolio_simulation"] = self._simulate_portfolio_performance(
            backtest_data
        )

        # 5. Calculate comprehensive risk metrics
        results["risk_metrics"] = self._calculate_risk_metrics(backtest_data)

        self.backtest_results = results

        self.logger.info("Comprehensive backtest completed")
        return results

    def _prepare_backtest_data(
        self, df: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Prepare and filter data for backtesting"""

        # Convert dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Filter by date range
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            mask = (df["Date"] >= start_dt) & (df["Date"] <= end_dt)
            filtered_df = df[mask].copy()
        else:
            filtered_df = df.copy()

        # Sort by date and symbol
        if all(col in filtered_df.columns for col in ["Symbol", "Date"]):
            filtered_df = filtered_df.sort_values(["Symbol", "Date"]).reset_index(
                drop=True
            )

        # Calculate additional metrics for backtesting
        if "Close" in filtered_df.columns and "Symbol" in filtered_df.columns:
            # Calculate forward returns for evaluation
            filtered_df["Forward_Return_1d"] = (
                filtered_df.groupby("Symbol")["Close"].pct_change().shift(-1)
            )
            filtered_df["Forward_Return_5d"] = (
                filtered_df.groupby("Symbol")["Close"].pct_change(5).shift(-5)
            )
            filtered_df["Forward_Return_20d"] = (
                filtered_df.groupby("Symbol")["Close"].pct_change(20).shift(-20)
            )

            # Calculate maximum drawdown in next N days
            for window in [5, 10, 20]:
                filtered_df[f"Max_Drawdown_{window}d"] = (
                    filtered_df.groupby("Symbol")["Close"]
                    .rolling(window)
                    .apply(lambda x: (x.min() - x.iloc[0]) / x.iloc[0])
                    .shift(-window)
                    .reset_index(level=0, drop=True)
                )

        return filtered_df

    def _get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics of backtest data"""

        summary = {
            "total_observations": len(df),
            "unique_symbols": df["Symbol"].nunique() if "Symbol" in df.columns else 0,
            "date_range": {
                "start": (
                    df["Date"].min().strftime("%Y-%m-%d")
                    if "Date" in df.columns
                    else None
                ),
                "end": (
                    df["Date"].max().strftime("%Y-%m-%d")
                    if "Date" in df.columns
                    else None
                ),
            },
            "missing_data_pct": (
                (df.isnull().sum().sum() / df.size * 100) if not df.empty else 0
            ),
        }

        # Risk label distribution
        if "Risk_Label" in df.columns:
            summary["risk_distribution"] = {
                "high_risk_periods": int(df["Risk_Label"].sum()),
                "total_periods": len(df),
                "risk_rate": float(df["Risk_Label"].mean()),
            }

        return summary

    def _analyze_market_events(
        self, df: pd.DataFrame, predictions_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Analyze system performance during major market events"""

        event_results = {}

        for event_name, event_info in self.market_events.items():
            self.logger.info(f"Analyzing {event_name}: {event_info['description']}")

            # Filter data for event period
            event_start = pd.to_datetime(event_info["start_date"])
            event_end = pd.to_datetime(event_info["end_date"])

            if "Date" not in df.columns:
                continue

            event_mask = (df["Date"] >= event_start) & (df["Date"] <= event_end)
            event_data = df[event_mask]

            if event_data.empty:
                event_results[event_name] = {
                    "status": "no_data",
                    "description": event_info["description"],
                }
                continue

            # Calculate event metrics
            event_analysis = {
                "description": event_info["description"],
                "type": event_info["type"],
                "period": f"{event_info['start_date']} to {event_info['end_date']}",
                "observations": len(event_data),
                "market_performance": {},
                "sentiment_analysis": {},
                "risk_signals": {},
                "early_warning_score": 0,
            }

            # Market performance during event
            if "Returns" in event_data.columns:
                returns = event_data["Returns"].dropna()
                event_analysis["market_performance"] = {
                    "avg_daily_return": float(returns.mean()),
                    "volatility": float(returns.std()),
                    "min_return": float(returns.min()),
                    "max_return": float(returns.max()),
                    "negative_days_pct": float((returns < 0).mean()),
                }

            # Sentiment analysis during event
            sentiment_cols = [
                col for col in event_data.columns if "sentiment" in col.lower()
            ]
            if sentiment_cols:
                event_analysis["sentiment_analysis"] = {}
                for col in sentiment_cols[:3]:  # Top 3 sentiment columns
                    sentiment_values = event_data[col].dropna()
                    if not sentiment_values.empty:
                        event_analysis["sentiment_analysis"][col] = {
                            "avg_sentiment": float(sentiment_values.mean()),
                            "sentiment_volatility": float(sentiment_values.std()),
                            "negative_sentiment_pct": float(
                                (sentiment_values < 0).mean()
                            ),
                        }

            # Risk signal analysis
            if "Risk_Label" in event_data.columns:
                risk_labels = event_data["Risk_Label"]
                event_analysis["risk_signals"] = {
                    "risk_periods": int(risk_labels.sum()),
                    "total_periods": len(risk_labels),
                    "risk_rate": float(risk_labels.mean()),
                    "avg_risk_score": float(
                        event_data.get("Risk_Score", pd.Series([0])).mean()
                    ),
                }

            # Early warning performance (did we predict this event?)
            pre_event_days = 30
            pre_event_start = event_start - timedelta(days=pre_event_days)
            pre_event_mask = (df["Date"] >= pre_event_start) & (
                df["Date"] < event_start
            )
            pre_event_data = df[pre_event_mask]

            if not pre_event_data.empty and "Risk_Label" in pre_event_data.columns:
                pre_event_risk_rate = pre_event_data["Risk_Label"].mean()
                baseline_risk_rate = (
                    df["Risk_Label"].mean() if "Risk_Label" in df.columns else 0
                )

                # Early warning score: how much higher was risk before the event
                early_warning_score = (pre_event_risk_rate - baseline_risk_rate) / (
                    baseline_risk_rate + 1e-6
                )
                event_analysis["early_warning_score"] = float(early_warning_score)

            # Model prediction analysis (if available)
            if predictions_df is not None and not predictions_df.empty:
                if (
                    "Date" in predictions_df.columns
                    and "Risk_Probability" in predictions_df.columns
                ):
                    pred_event_mask = (predictions_df["Date"] >= event_start) & (
                        predictions_df["Date"] <= event_end
                    )
                    pred_event_data = predictions_df[pred_event_mask]

                    if not pred_event_data.empty:
                        event_analysis["prediction_performance"] = {
                            "avg_risk_probability": float(
                                pred_event_data["Risk_Probability"].mean()
                            ),
                            "max_risk_probability": float(
                                pred_event_data["Risk_Probability"].max()
                            ),
                            "high_risk_predictions_pct": float(
                                (pred_event_data["Risk_Probability"] > 0.7).mean()
                            ),
                        }

            event_results[event_name] = event_analysis

        return event_results

    def _calculate_prediction_accuracy(
        self, df: pd.DataFrame, predictions_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate prediction accuracy metrics"""

        if predictions_df.empty or "Risk_Label" not in df.columns:
            return {}

        # Merge actual labels with predictions
        if all(col in df.columns for col in ["Symbol", "Date", "Risk_Label"]) and all(
            col in predictions_df.columns
            for col in ["Symbol", "Date", "Risk_Probability"]
        ):

            merged_data = df[["Symbol", "Date", "Risk_Label"]].merge(
                predictions_df[["Symbol", "Date", "Risk_Probability"]],
                on=["Symbol", "Date"],
                how="inner",
            )
        else:
            return {}

        if merged_data.empty:
            return {}

        y_true = merged_data["Risk_Label"].values
        y_prob = merged_data["Risk_Probability"].values

        # Calculate metrics for different thresholds
        thresholds = [0.3, 0.5, 0.7]
        accuracy_results = {}

        for threshold in thresholds:
            y_pred = (y_prob > threshold).astype(int)

            # Basic metrics
            accuracy = (y_pred == y_true).mean()

            # Confusion matrix elements
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            accuracy_results[f"threshold_{threshold}"] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score),
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
            }

        # AUC calculation
        try:
            from sklearn.metrics import roc_auc_score

            auc_score = roc_auc_score(y_true, y_prob)
            accuracy_results["auc_score"] = float(auc_score)
        except:
            accuracy_results["auc_score"] = None

        return accuracy_results

    def _evaluate_early_warning(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate early warning capabilities"""

        if "Risk_Label" not in df.columns or "Symbol" not in df.columns:
            return {}

        early_warning_results = {
            "lead_time_analysis": {},
            "signal_persistence": {},
            "false_alarm_rate": {},
            "coverage_analysis": {},
        }

        # Lead time analysis: How many days before actual risk events do we signal?
        lead_times = []

        for symbol in df["Symbol"].unique():
            symbol_data = df[df["Symbol"] == symbol].sort_values("Date")

            # Find actual risk events (consecutive high-risk periods)
            risk_events = symbol_data["Risk_Label"] == 1
            risk_event_starts = risk_events & ~risk_events.shift(1, fill_value=False)

            for idx in symbol_data[risk_event_starts].index:
                # Look back to find first warning signal
                lookback_data = symbol_data[symbol_data.index < idx].tail(
                    30
                )  # 30 days lookback

                if "Risk_Score" in lookback_data.columns:
                    warning_signals = lookback_data["Risk_Score"] > 0.5
                    if warning_signals.any():
                        first_warning_idx = lookback_data[warning_signals].index[0]
                        lead_time = idx - first_warning_idx
                        lead_times.append(lead_time)

        if lead_times:
            early_warning_results["lead_time_analysis"] = {
                "avg_lead_time_days": float(np.mean(lead_times)),
                "median_lead_time_days": float(np.median(lead_times)),
                "max_lead_time_days": int(np.max(lead_times)),
                "min_lead_time_days": int(np.min(lead_times)),
            }

        # Signal persistence: How long do warning signals last?
        if "Risk_Score" in df.columns:
            warning_signals = df["Risk_Score"] > 0.5

            # Calculate consecutive warning periods
            signal_groups = (warning_signals != warning_signals.shift()).cumsum()
            warning_periods = (
                df[warning_signals].groupby([df["Symbol"], signal_groups]).size()
            )

            if not warning_periods.empty:
                early_warning_results["signal_persistence"] = {
                    "avg_signal_duration_days": float(warning_periods.mean()),
                    "median_signal_duration_days": float(warning_periods.median()),
                    "max_signal_duration_days": int(warning_periods.max()),
                }

        # False alarm rate: Warnings not followed by actual risk events
        false_alarms = 0
        total_warnings = 0

        for symbol in df["Symbol"].unique():
            symbol_data = df[df["Symbol"] == symbol].sort_values("Date")

            if "Risk_Score" in symbol_data.columns:
                warning_signals = symbol_data["Risk_Score"] > 0.5

                for idx in symbol_data[warning_signals].index:
                    total_warnings += 1

                    # Check if followed by actual risk event within 30 days
                    future_data = symbol_data[symbol_data.index > idx].head(30)
                    if not (future_data["Risk_Label"] == 1).any():
                        false_alarms += 1

        if total_warnings > 0:
            early_warning_results["false_alarm_rate"] = {
                "false_alarms": false_alarms,
                "total_warnings": total_warnings,
                "false_alarm_rate": false_alarms / total_warnings,
            }

        return early_warning_results

    def _simulate_portfolio_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Simulate portfolio performance using risk signals"""

        if not all(col in df.columns for col in ["Symbol", "Date", "Returns"]):
            return {}

        # Simple trading strategy based on risk signals
        portfolio_results = {
            "strategy_performance": {},
            "risk_adjusted_returns": {},
            "drawdown_analysis": {},
        }

        # Strategy 1: Reduce exposure when risk signals are high
        if "Risk_Score" in df.columns:
            daily_returns = (
                df.groupby("Date")
                .agg(
                    {
                        "Returns": "mean",  # Market return
                        "Risk_Score": "mean",  # Average risk score
                    }
                )
                .reset_index()
            )

            # Risk-adjusted position sizing
            daily_returns["Position_Size"] = 1.0 - (
                daily_returns["Risk_Score"] * 0.8
            )  # Reduce exposure when risk is high
            daily_returns["Position_Size"] = daily_returns["Position_Size"].clip(
                0.2, 1.0
            )  # Min 20% exposure

            # Calculate strategy returns
            daily_returns["Strategy_Return"] = (
                daily_returns["Returns"] * daily_returns["Position_Size"]
            )
            daily_returns["Benchmark_Return"] = daily_returns["Returns"]  # Buy and hold

            # Cumulative returns
            daily_returns["Strategy_Cumulative"] = (
                1 + daily_returns["Strategy_Return"]
            ).cumprod()
            daily_returns["Benchmark_Cumulative"] = (
                1 + daily_returns["Benchmark_Return"]
            ).cumprod()

            # Performance metrics
            total_periods = len(daily_returns)
            if total_periods > 0:
                strategy_total_return = (
                    daily_returns["Strategy_Cumulative"].iloc[-1] - 1
                )
                benchmark_total_return = (
                    daily_returns["Benchmark_Cumulative"].iloc[-1] - 1
                )

                strategy_volatility = daily_returns["Strategy_Return"].std() * np.sqrt(
                    252
                )
                benchmark_volatility = daily_returns[
                    "Benchmark_Return"
                ].std() * np.sqrt(252)

                # Sharpe ratio (assuming 0% risk-free rate)
                strategy_sharpe = (
                    (daily_returns["Strategy_Return"].mean() * 252)
                    / strategy_volatility
                    if strategy_volatility > 0
                    else 0
                )
                benchmark_sharpe = (
                    (daily_returns["Benchmark_Return"].mean() * 252)
                    / benchmark_volatility
                    if benchmark_volatility > 0
                    else 0
                )

                # Maximum drawdown
                strategy_peak = daily_returns["Strategy_Cumulative"].expanding().max()
                strategy_drawdown = (
                    daily_returns["Strategy_Cumulative"] - strategy_peak
                ) / strategy_peak
                max_drawdown = strategy_drawdown.min()

                portfolio_results["strategy_performance"] = {
                    "total_return": float(strategy_total_return),
                    "benchmark_return": float(benchmark_total_return),
                    "excess_return": float(
                        strategy_total_return - benchmark_total_return
                    ),
                    "volatility": float(strategy_volatility),
                    "sharpe_ratio": float(strategy_sharpe),
                    "max_drawdown": float(max_drawdown),
                    "benchmark_sharpe": float(benchmark_sharpe),
                }

        return portfolio_results

    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""

        risk_metrics = {
            "market_risk_metrics": {},
            "model_stability": {},
            "feature_stability": {},
        }

        # Market risk metrics
        if "Returns" in df.columns:
            returns = df["Returns"].dropna()

            if not returns.empty:
                # Value at Risk (VaR) at different confidence levels
                var_95 = returns.quantile(0.05)
                var_99 = returns.quantile(0.01)

                # Expected Shortfall (Conditional VaR)
                es_95 = returns[returns <= var_95].mean()
                es_99 = returns[returns <= var_99].mean()

                risk_metrics["market_risk_metrics"] = {
                    "var_95": float(var_95),
                    "var_99": float(var_99),
                    "expected_shortfall_95": float(es_95),
                    "expected_shortfall_99": float(es_99),
                    "volatility_annualized": float(returns.std() * np.sqrt(252)),
                    "skewness": float(returns.skew()),
                    "kurtosis": float(returns.kurtosis()),
                }

        # Model stability over time
        if "Risk_Score" in df.columns and "Date" in df.columns:
            # Calculate rolling statistics of risk scores
            df_sorted = df.sort_values("Date")
            rolling_window = 90  # 3 months

            df_sorted["Risk_Score_MA"] = (
                df_sorted["Risk_Score"].rolling(rolling_window).mean()
            )
            df_sorted["Risk_Score_Vol"] = (
                df_sorted["Risk_Score"].rolling(rolling_window).std()
            )

            risk_metrics["model_stability"] = {
                "avg_risk_score": float(df_sorted["Risk_Score"].mean()),
                "risk_score_volatility": float(df_sorted["Risk_Score"].std()),
                "avg_rolling_volatility": float(df_sorted["Risk_Score_Vol"].mean()),
                "model_drift": float(
                    df_sorted["Risk_Score_MA"].iloc[-1]
                    - df_sorted["Risk_Score_MA"].iloc[rolling_window]
                ),
            }

        return risk_metrics

    def generate_backtest_report(self, output_dir: str) -> str:
        """Generate comprehensive backtest report"""

        if not self.backtest_results:
            self.logger.error("No backtest results available")
            return ""

        os.makedirs(output_dir, exist_ok=True)

        # Save detailed results as JSON
        results_path = os.path.join(output_dir, "backtest_results.json")
        with open(results_path, "w") as f:
            json.dump(self.backtest_results, f, indent=2, default=str)

        # Generate HTML report
        html_report = self._create_html_backtest_report()

        report_path = os.path.join(output_dir, "backtest_report.html")
        with open(report_path, "w") as f:
            f.write(html_report)

        self.logger.info(f"Generated backtest report: {report_path}")
        return report_path

    def _create_html_backtest_report(self) -> str:
        """Create HTML backtest report"""

        results = self.backtest_results

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Market Risk System Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 15px; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-top: 30px; }}
                h3 {{ color: #34495e; margin-top: 25px; }}
                .summary-box {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; margin: 15px 0; }}
                .metric {{ display: inline-block; margin: 10px 15px; }}
                .metric-value {{ font-weight: bold; color: #2c3e50; }}
                .event-analysis {{ background: #fff3cd; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .performance-good {{ color: #28a745; font-weight: bold; }}
                .performance-poor {{ color: #dc3545; font-weight: bold; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .timestamp {{ text-align: center; color: #999; margin-top: 40px; }}
            </style>
        </head>
        <body>
            <h1>Market Risk Early Warning System - Backtest Report</h1>
        """

        # Executive Summary
        html_content += f"""
        <div class="summary-box">
            <h2>Executive Summary</h2>
            <p><strong>Backtest Period:</strong> {results.get('backtest_period', {}).get('start', 'N/A')} to {results.get('backtest_period', {}).get('end', 'N/A')}</p>
            <div class="metric">
                <span>Total Observations:</span> 
                <span class="metric-value">{results.get('data_summary', {}).get('total_observations', 0):,}</span>
            </div>
            <div class="metric">
                <span>Unique Symbols:</span>
                <span class="metric-value">{results.get('data_summary', {}).get('unique_symbols', 0)}</span>
            </div>
            <div class="metric">
                <span>Risk Rate:</span>
                <span class="metric-value">{results.get('data_summary', {}).get('risk_distribution', {}).get('risk_rate', 0):.2%}</span>
            </div>
        </div>
        """

        # Event Analysis
        html_content += "<h2>Major Market Events Analysis</h2>"

        event_analysis = results.get("event_analysis", {})
        for event_name, event_data in event_analysis.items():
            if isinstance(event_data, dict) and event_data.get("status") != "no_data":
                html_content += f"""
                <div class="event-analysis">
                    <h3>{event_data.get('description', event_name)}</h3>
                    <p><strong>Period:</strong> {event_data.get('period', 'N/A')}</p>
                    <p><strong>Type:</strong> {event_data.get('type', 'N/A')}</p>
                """

                # Market performance
                market_perf = event_data.get("market_performance", {})
                if market_perf:
                    html_content += f"""
                    <p><strong>Market Performance:</strong></p>
                    <ul>
                        <li>Average Daily Return: <span class="metric-value">{market_perf.get('avg_daily_return', 0):.2%}</span></li>
                        <li>Volatility: <span class="metric-value">{market_perf.get('volatility', 0):.2%}</span></li>
                        <li>Negative Days: <span class="metric-value">{market_perf.get('negative_days_pct', 0):.1%}</span></li>
                    </ul>
                    """

                # Risk signals
                risk_signals = event_data.get("risk_signals", {})
                if risk_signals:
                    risk_rate = risk_signals.get("risk_rate", 0)
                    performance_class = (
                        "performance-good" if risk_rate > 0.3 else "performance-poor"
                    )

                    html_content += f"""
                    <p><strong>Risk Detection:</strong></p>
                    <ul>
                        <li>Risk Rate During Event: <span class="{performance_class}">{risk_rate:.1%}</span></li>
                        <li>Average Risk Score: <span class="metric-value">{risk_signals.get('avg_risk_score', 0):.2f}</span></li>
                    </ul>
                    """

                # Early warning score
                early_warning = event_data.get("early_warning_score", 0)
                warning_class = (
                    "performance-good" if early_warning > 0.5 else "performance-poor"
                )
                html_content += f"<p><strong>Early Warning Score:</strong> <span class='{warning_class}'>{early_warning:.2f}</span></p>"

                html_content += "</div>"

        # Prediction Accuracy
        pred_accuracy = results.get("prediction_accuracy", {})
        if pred_accuracy:
            html_content += """
            <h2>Prediction Accuracy</h2>
            <table>
                <tr><th>Threshold</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th></tr>
            """

            for threshold_key, metrics in pred_accuracy.items():
                if threshold_key.startswith("threshold_"):
                    threshold = threshold_key.split("_")[1]
                    html_content += f"""
                    <tr>
                        <td>{threshold}</td>
                        <td>{metrics.get('accuracy', 0):.3f}</td>
                        <td>{metrics.get('precision', 0):.3f}</td>
                        <td>{metrics.get('recall', 0):.3f}</td>
                        <td>{metrics.get('f1_score', 0):.3f}</td>
                    </tr>
                    """

            html_content += "</table>"

            auc_score = pred_accuracy.get("auc_score")
            if auc_score is not None:
                auc_class = (
                    "performance-good" if auc_score > 0.7 else "performance-poor"
                )
                html_content += f"<p><strong>AUC Score:</strong> <span class='{auc_class}'>{auc_score:.3f}</span></p>"

        # Portfolio Performance
        portfolio_perf = results.get("portfolio_simulation", {}).get(
            "strategy_performance", {}
        )
        if portfolio_perf:
            html_content += f"""
            <h2>Portfolio Simulation Results</h2>
            <div class="summary-box">
                <div class="metric">
                    <span>Strategy Return:</span>
                    <span class="metric-value">{portfolio_perf.get('total_return', 0):.2%}</span>
                </div>
                <div class="metric">
                    <span>Benchmark Return:</span>
                    <span class="metric-value">{portfolio_perf.get('benchmark_return', 0):.2%}</span>
                </div>
                <div class="metric">
                    <span>Excess Return:</span>
                    <span class="metric-value">{portfolio_perf.get('excess_return', 0):.2%}</span>
                </div>
                <div class="metric">
                    <span>Sharpe Ratio:</span>
                    <span class="metric-value">{portfolio_perf.get('sharpe_ratio', 0):.2f}</span>
                </div>
                <div class="metric">
                    <span>Max Drawdown:</span>
                    <span class="metric-value">{portfolio_perf.get('max_drawdown', 0):.2%}</span>
                </div>
            </div>
            """

        html_content += f"""
            <div class="timestamp">
                <p>Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """

        return html_content

    def save_backtest_data(self, output_dir: str):
        """Save backtest data for further analysis"""

        os.makedirs(output_dir, exist_ok=True)

        # Save results as JSON
        if self.backtest_results:
            results_path = os.path.join(output_dir, "detailed_backtest_results.json")
            with open(results_path, "w") as f:
                json.dump(self.backtest_results, f, indent=2, default=str)

            self.logger.info(f"Saved backtest data to {output_dir}")
