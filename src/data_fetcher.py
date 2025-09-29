"""
Data fetcher module for stock prices and financial ratios
Handles Yahoo Finance and Alpha Vantage APIs with rate limiting
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os

class StockDataFetcher:
    """Fetches stock data from Yahoo Finance and Alpha Vantage"""
    
    def __init__(self, alpha_vantage_key: str = None):
        self.alpha_vantage_key = alpha_vantage_key
        self.logger = logging.getLogger(__name__)
        self.rate_limit_delay = 12  # seconds between Alpha Vantage calls (5 per minute limit)
        
    def fetch_yahoo_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        self.logger.info(f"Fetching Yahoo Finance data for {len(symbols)} symbols")
        stock_data = {}
        
        for symbol in symbols:
            try:
                self.logger.info(f"Fetching data for {symbol}")
                
                # Fetch stock data
                ticker = yf.Ticker(symbol)
                hist_data = ticker.history(start=start_date, end=end_date)
                
                if hist_data.empty:
                    self.logger.warning(f"No data found for {symbol}")
                    continue
                
                # Get additional info
                info = ticker.info
                
                # Calculate technical indicators
                hist_data['Returns'] = hist_data['Close'].pct_change()
                hist_data['Volatility_30d'] = hist_data['Returns'].rolling(window=30).std() * np.sqrt(252)
                hist_data['MA_50'] = hist_data['Close'].rolling(window=50).mean()
                hist_data['MA_200'] = hist_data['Close'].rolling(window=200).mean()
                hist_data['RSI'] = self._calculate_rsi(hist_data['Close'])
                
                # Add fundamental data
                hist_data['PE_Ratio'] = info.get('trailingPE', np.nan)
                hist_data['Debt_to_Equity'] = info.get('debtToEquity', np.nan)
                hist_data['ROE'] = info.get('returnOnEquity', np.nan)
                hist_data['Market_Cap'] = info.get('marketCap', np.nan)
                hist_data['Beta'] = info.get('beta', np.nan)
                
                # Add symbol column
                hist_data['Symbol'] = symbol
                hist_data.reset_index(inplace=True)
                
                stock_data[symbol] = hist_data
                
                # Rate limiting for Yahoo Finance
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
                continue
                
        return stock_data
    
    def fetch_alpha_vantage_fundamentals(self, symbol: str) -> Dict:
        """
        Fetch fundamental data from Alpha Vantage
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with fundamental data
        """
        if not self.alpha_vantage_key:
            self.logger.warning("Alpha Vantage API key not provided")
            return {}
            
        try:
            # Company Overview
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'Error Message' in data or 'Note' in data:
                self.logger.warning(f"Alpha Vantage error for {symbol}: {data}")
                return {}
                
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching Alpha Vantage data for {symbol}: {str(e)}")
            return {}
    
    def fetch_earnings_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch earnings data from Alpha Vantage
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with earnings data
        """
        if not self.alpha_vantage_key:
            return pd.DataFrame()
            
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'EARNINGS',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'Error Message' in data or 'Note' in data:
                return pd.DataFrame()
                
            # Parse quarterly earnings
            quarterly_earnings = data.get('quarterlyEarnings', [])
            if quarterly_earnings:
                df = pd.DataFrame(quarterly_earnings)
                df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
                df['Symbol'] = symbol
                return df
                
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error fetching earnings for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def fetch_sp500_list(self) -> List[str]:
        """
        Fetch current S&P 500 list from Wikipedia
        
        Returns:
            List of S&P 500 symbols
        """
        try:
            # Read S&P 500 list from Wikipedia
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500_table = tables[0]
            symbols = sp500_table['Symbol'].tolist()
            
            # Clean symbols (remove dots and special characters)
            symbols = [symbol.replace('.', '-') for symbol in symbols]
            
            self.logger.info(f"Fetched {len(symbols)} S&P 500 symbols")
            return symbols
            
        except Exception as e:
            self.logger.error(f"Error fetching S&P 500 list: {str(e)}")
            # Return default list if Wikipedia fails
            return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 'UNH', 'JNJ',
                'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'BAC', 'ABBV', 'PFE'
            ]
    
    def calculate_financial_ratios(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Calculate additional financial ratios
        
        Args:
            stock_data: Dictionary of stock DataFrames
            
        Returns:
            Updated dictionary with additional ratios
        """
        for symbol, df in stock_data.items():
            try:
                # Price-to-Book ratio (if book value available)
                if 'Book_Value' in df.columns:
                    df['PB_Ratio'] = df['Close'] / df['Book_Value']
                
                # Price momentum indicators
                df['Price_Change_1d'] = df['Close'].pct_change(1)
                df['Price_Change_5d'] = df['Close'].pct_change(5)
                df['Price_Change_20d'] = df['Close'].pct_change(20)
                
                # Volume indicators
                df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
                df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
                
                # Volatility measures
                df['High_Low_Ratio'] = df['High'] / df['Low']
                df['Close_to_High'] = df['Close'] / df['High']
                
                # Trend indicators
                df['MA_Cross'] = (df['MA_50'] > df['MA_200']).astype(int)
                
            except Exception as e:
                self.logger.error(f"Error calculating ratios for {symbol}: {str(e)}")
                
        return stock_data
    
    def save_data(self, stock_data: Dict[str, pd.DataFrame], output_dir: str):
        """
        Save stock data to files
        
        Args:
            stock_data: Dictionary of stock DataFrames
            output_dir: Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual stock files
        for symbol, df in stock_data.items():
            file_path = os.path.join(output_dir, f"{symbol}_stock_data.csv")
            df.to_csv(file_path, index=False)
        
        # Save combined data
        combined_df = pd.concat(stock_data.values(), ignore_index=True)
        combined_path = os.path.join(output_dir, "combined_stock_data.csv")
        combined_df.to_csv(combined_path, index=False)
        
        self.logger.info(f"Saved data for {len(stock_data)} symbols to {output_dir}")
    
    def fetch_all_data(self, symbols: List[str], start_date: str, end_date: str, 
                       output_dir: str, use_alpha_vantage: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch all available data for given symbols
        
        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            output_dir: Directory to save data
            use_alpha_vantage: Whether to use Alpha Vantage for additional data
            
        Returns:
            Dictionary with all stock data
        """
        # Fetch Yahoo Finance data
        stock_data = self.fetch_yahoo_data(symbols, start_date, end_date)
        
        # Enhance with Alpha Vantage data if available
        if use_alpha_vantage and self.alpha_vantage_key:
            self.logger.info("Enhancing with Alpha Vantage fundamental data")
            
            for symbol in symbols:
                if symbol in stock_data:
                    # Get fundamental data
                    fundamentals = self.fetch_alpha_vantage_fundamentals(symbol)
                    
                    if fundamentals:
                        # Add fundamental metrics to DataFrame
                        df = stock_data[symbol]
                        df['Revenue_TTM'] = fundamentals.get('RevenueTTM', np.nan)
                        df['Gross_Profit_TTM'] = fundamentals.get('GrossProfitTTM', np.nan)
                        df['EBITDA'] = fundamentals.get('EBITDA', np.nan)
                        df['Operating_Margin_TTM'] = fundamentals.get('OperatingMarginTTM', np.nan)
                        df['Profit_Margin'] = fundamentals.get('ProfitMargin', np.nan)
                        df['Forward_PE'] = fundamentals.get('ForwardPE', np.nan)
                        df['PEG_Ratio'] = fundamentals.get('PEGRatio', np.nan)
                    
                    # Rate limiting for Alpha Vantage
                    time.sleep(self.rate_limit_delay)
        
        # Calculate additional ratios
        stock_data = self.calculate_financial_ratios(stock_data)
        
        # Save data
        self.save_data(stock_data, output_dir)
        
        return stock_data
