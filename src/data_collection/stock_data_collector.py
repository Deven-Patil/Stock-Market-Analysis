"""
Stock Data Collector Module

This module provides functionality to collect stock market data from various sources
including Yahoo Finance and Alpha Vantage APIs.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import requests
import time
from datetime import datetime, timedelta
import os
from typing import Optional, Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataCollector:
    """
    A class to collect stock market data from various sources.
    """
    
    def __init__(self, alpha_vantage_key: Optional[str] = None):
        """
        Initialize the StockDataCollector.
        
        Args:
            alpha_vantage_key (str, optional): API key for Alpha Vantage
        """
        self.alpha_vantage_key = alpha_vantage_key or os.getenv('ALPHA_VANTAGE_KEY')
        if self.alpha_vantage_key:
            self.ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
        
    def get_yahoo_finance_data(self, 
                              symbol: str, 
                              start_date: str = None, 
                              end_date: str = None,
                              period: str = "1y") -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'MSFT')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            period (str): Time period if start/end dates not provided
            
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        try:
            logger.info(f"Fetching data for {symbol} from Yahoo Finance")
            
            ticker = yf.Ticker(symbol)
            
            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date)
            else:
                data = ticker.history(period=period)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Reset index to make date a column
            data.reset_index(inplace=True)
            data.rename(columns={'Date': 'date'}, inplace=True)
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_alpha_vantage_data(self, 
                              symbol: str, 
                              interval: str = 'daily',
                              outputsize: str = 'full') -> pd.DataFrame:
        """
        Fetch stock data from Alpha Vantage API.
        
        Args:
            symbol (str): Stock symbol
            interval (str): Time interval ('1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly')
            outputsize (str): Output size ('compact' or 'full')
            
        Returns:
            pd.DataFrame: Stock data
        """
        if not self.alpha_vantage_key:
            logger.error("Alpha Vantage API key not provided")
            return pd.DataFrame()
        
        try:
            logger.info(f"Fetching data for {symbol} from Alpha Vantage")
            
            if interval == 'daily':
                data, meta_data = self.ts.get_daily(symbol=symbol, outputsize=outputsize)
            elif interval == 'weekly':
                data, meta_data = self.ts.get_weekly(symbol=symbol)
            elif interval == 'monthly':
                data, meta_data = self.ts.get_monthly(symbol=symbol)
            else:
                logger.error(f"Unsupported interval: {interval}")
                return pd.DataFrame()
            
            # Reset index to make date a column
            data.reset_index(inplace=True)
            data.rename(columns={'date': 'date'}, inplace=True)
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_multiple_stocks(self, 
                           symbols: List[str], 
                           source: str = 'yahoo',
                           **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks.
        
        Args:
            symbols (List[str]): List of stock symbols
            source (str): Data source ('yahoo' or 'alpha_vantage')
            **kwargs: Additional arguments for data fetching
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with symbol as key and data as value
        """
        data_dict = {}
        
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}")
            
            if source.lower() == 'yahoo':
                data = self.get_yahoo_finance_data(symbol, **kwargs)
            elif source.lower() == 'alpha_vantage':
                data = self.get_alpha_vantage_data(symbol, **kwargs)
            else:
                logger.error(f"Unsupported source: {source}")
                continue
            
            if not data.empty:
                data_dict[symbol] = data
            
            # Add delay to avoid rate limiting
            time.sleep(1)
        
        return data_dict
    
    def save_data(self, 
                  data: pd.DataFrame, 
                  symbol: str, 
                  format: str = 'csv',
                  directory: str = 'data') -> str:
        """
        Save stock data to file.
        
        Args:
            data (pd.DataFrame): Stock data to save
            symbol (str): Stock symbol
            format (str): File format ('csv', 'parquet', 'json')
            directory (str): Directory to save the file
            
        Returns:
            str: Path to saved file
        """
        os.makedirs(directory, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_{timestamp}.{format}"
        filepath = os.path.join(directory, filename)
        
        try:
            if format.lower() == 'csv':
                data.to_csv(filepath, index=False)
            elif format.lower() == 'parquet':
                data.to_parquet(filepath, index=False)
            elif format.lower() == 'json':
                data.to_json(filepath, orient='records', date_format='iso')
            else:
                logger.error(f"Unsupported format: {format}")
                return ""
            
            logger.info(f"Data saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return ""
    
    def load_data(self, 
                  filepath: str, 
                  format: str = 'csv') -> pd.DataFrame:
        """
        Load stock data from file.
        
        Args:
            filepath (str): Path to the data file
            format (str): File format ('csv', 'parquet', 'json')
            
        Returns:
            pd.DataFrame: Loaded stock data
        """
        try:
            if format.lower() == 'csv':
                data = pd.read_csv(filepath)
            elif format.lower() == 'parquet':
                data = pd.read_parquet(filepath)
            elif format.lower() == 'json':
                data = pd.read_json(filepath)
            else:
                logger.error(f"Unsupported format: {format}")
                return pd.DataFrame()
            
            # Convert date column to datetime
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            
            logger.info(f"Data loaded from {filepath}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()

# Example usage and testing
if __name__ == "__main__":
    # Initialize collector
    collector = StockDataCollector()
    
    # Fetch data for Apple stock
    apple_data = collector.get_yahoo_finance_data('AAPL', period='1y')
    
    if not apple_data.empty:
        print(f"Fetched {len(apple_data)} records for AAPL")
        print(apple_data.head())
        
        # Save data
        collector.save_data(apple_data, 'AAPL', 'csv') 