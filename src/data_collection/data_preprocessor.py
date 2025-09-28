"""
Data Preprocessing Module

This module provides functionality to clean, transform, and prepare stock market data
for time series analysis and modeling.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
import logging
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataPreprocessor:
    """
    A class to preprocess stock market data for analysis.
    """
    
    def __init__(self):
        """Initialize the StockDataPreprocessor."""
        pass
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean stock data by handling missing values, outliers, and data types.
        
        Args:
            data (pd.DataFrame): Raw stock data
            
        Returns:
            pd.DataFrame: Cleaned stock data
        """
        logger.info("Starting data cleaning process")
        
        # Make a copy to avoid modifying original data
        cleaned_data = data.copy()
        
        # Convert date column to datetime if it exists
        if 'date' in cleaned_data.columns:
            cleaned_data['date'] = pd.to_datetime(cleaned_data['date'])
        
        # Handle missing values
        cleaned_data = self._handle_missing_values(cleaned_data)
        
        # Remove outliers
        cleaned_data = self._remove_outliers(cleaned_data)
        
        # Sort by date
        if 'date' in cleaned_data.columns:
            cleaned_data = cleaned_data.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Data cleaning completed. Shape: {cleaned_data.shape}")
        return cleaned_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with handled missing values
        """
        # Check for missing values
        missing_counts = data.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Missing values found: {missing_counts[missing_counts > 0]}")
        
        # For OHLCV data, forward fill is often appropriate
        ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col for col in ohlcv_columns if col in data.columns]
        
        if available_columns:
            # Forward fill for OHLCV data
            data[available_columns] = data[available_columns].fillna(method='ffill')
            # Backward fill for any remaining missing values at the beginning
            data[available_columns] = data[available_columns].fillna(method='bfill')
        
        # For other numeric columns, use interpolation
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in data.columns and data[col].isnull().sum() > 0:
                data[col] = data[col].interpolate(method='linear')
        
        return data
    
    def _remove_outliers(self, data: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Remove outliers from the dataset.
        
        Args:
            data (pd.DataFrame): Input data
            method (str): Method to detect outliers ('iqr', 'zscore')
            
        Returns:
            pd.DataFrame: Data with outliers removed
        """
        ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col for col in ohlcv_columns if col in data.columns]
        
        if not available_columns:
            return data
        
        original_shape = data.shape
        
        for col in available_columns:
            if col == 'Volume':
                # For volume, use different outlier detection
                continue
            
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Remove outliers
                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
            
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data[col]))
                data = data[z_scores < 3]  # Remove points with z-score > 3
        
        logger.info(f"Outliers removed. Shape changed from {original_shape} to {data.shape}")
        return data
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataset.
        
        Args:
            data (pd.DataFrame): Stock data
            
        Returns:
            pd.DataFrame: Data with technical indicators
        """
        logger.info("Adding technical indicators")
        
        if 'Close' not in data.columns:
            logger.warning("Close price not found. Cannot add technical indicators.")
            return data
        
        # Moving averages
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # RSI
        data['RSI'] = self._calculate_rsi(data['Close'])
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        # Price changes
        data['Price_Change'] = data['Close'].pct_change()
        data['Price_Change_5d'] = data['Close'].pct_change(periods=5)
        data['Price_Change_20d'] = data['Close'].pct_change(periods=20)
        
        # Volatility
        data['Volatility'] = data['Price_Change'].rolling(window=20).std()
        
        # Volume indicators
        if 'Volume' in data.columns:
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        logger.info("Technical indicators added successfully")
        return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices (pd.Series): Price series
            period (int): RSI period
            
        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features for modeling.
        
        Args:
            data (pd.DataFrame): Stock data
            
        Returns:
            pd.DataFrame: Data with additional features
        """
        logger.info("Creating additional features")
        
        if 'date' not in data.columns:
            logger.warning("Date column not found. Cannot create time-based features.")
            return data
        
        # Time-based features
        data['Year'] = data['date'].dt.year
        data['Month'] = data['date'].dt.month
        data['Day'] = data['date'].dt.day
        data['DayOfWeek'] = data['date'].dt.dayofweek
        data['Quarter'] = data['date'].dt.quarter
        
        # Lag features
        if 'Close' in data.columns:
            data['Close_Lag1'] = data['Close'].shift(1)
            data['Close_Lag5'] = data['Close'].shift(5)
            data['Close_Lag20'] = data['Close'].shift(20)
        
        # Rolling statistics
        if 'Close' in data.columns:
            data['Close_Rolling_Mean_5'] = data['Close'].rolling(window=5).mean()
            data['Close_Rolling_Std_5'] = data['Close'].rolling(window=5).std()
            data['Close_Rolling_Min_5'] = data['Close'].rolling(window=5).min()
            data['Close_Rolling_Max_5'] = data['Close'].rolling(window=5).max()
        
        logger.info("Additional features created successfully")
        return data
    
    def prepare_for_modeling(self, 
                           data: pd.DataFrame, 
                           target_column: str = 'Close',
                           feature_columns: Optional[List[str]] = None,
                           test_size: float = 0.2,
                           validation_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for modeling by splitting into train/validation/test sets.
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Target variable column
            feature_columns (List[str], optional): Feature columns to use
            test_size (float): Proportion of data for test set
            validation_size (float): Proportion of data for validation set
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test sets
        """
        logger.info("Preparing data for modeling")
        
        # Remove rows with missing values
        data_clean = data.dropna()
        
        if data_clean.empty:
            logger.error("No data remaining after removing missing values")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Define feature columns if not provided
        if feature_columns is None:
            # Exclude date and target columns from features
            exclude_columns = ['date', target_column]
            feature_columns = [col for col in data_clean.columns if col not in exclude_columns]
        
        # Ensure all required columns exist
        required_columns = [target_column] + feature_columns
        missing_columns = [col for col in required_columns if col not in data_clean.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Select only required columns
        data_modeling = data_clean[required_columns].copy()
        
        # Split data chronologically
        n = len(data_modeling)
        test_start = int(n * (1 - test_size - validation_size))
        val_start = int(n * (1 - test_size))
        
        train_data = data_modeling.iloc[:test_start]
        val_data = data_modeling.iloc[test_start:val_start]
        test_data = data_modeling.iloc[val_start:]
        
        logger.info(f"Data split - Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def normalize_data(self, 
                      data: pd.DataFrame, 
                      method: str = 'minmax',
                      feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normalize numerical features.
        
        Args:
            data (pd.DataFrame): Input data
            method (str): Normalization method ('minmax', 'standard', 'robust')
            feature_columns (List[str], optional): Columns to normalize
            
        Returns:
            pd.DataFrame: Normalized data
        """
        logger.info(f"Normalizing data using {method} method")
        
        if feature_columns is None:
            # Select only numeric columns
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        normalized_data = data.copy()
        
        for col in feature_columns:
            if col in data.columns:
                if method == 'minmax':
                    min_val = data[col].min()
                    max_val = data[col].max()
                    if max_val != min_val:
                        normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
                
                elif method == 'standard':
                    mean_val = data[col].mean()
                    std_val = data[col].std()
                    if std_val != 0:
                        normalized_data[col] = (data[col] - mean_val) / std_val
                
                elif method == 'robust':
                    median_val = data[col].median()
                    q75 = data[col].quantile(0.75)
                    q25 = data[col].quantile(0.25)
                    iqr = q75 - q25
                    if iqr != 0:
                        normalized_data[col] = (data[col] - median_val) / iqr
        
        logger.info("Data normalization completed")
        return normalized_data

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'Open': np.random.uniform(100, 200, 100),
        'High': np.random.uniform(200, 300, 100),
        'Low': np.random.uniform(50, 100, 100),
        'Close': np.random.uniform(100, 200, 100),
        'Volume': np.random.uniform(1000000, 5000000, 100)
    })
    
    # Initialize preprocessor
    preprocessor = StockDataPreprocessor()
    
    # Clean data
    cleaned_data = preprocessor.clean_data(sample_data)
    
    # Add technical indicators
    data_with_indicators = preprocessor.add_technical_indicators(cleaned_data)
    
    # Create features
    final_data = preprocessor.create_features(data_with_indicators)
    
    print("Preprocessing completed!")
    print(f"Final data shape: {final_data.shape}")
    print(f"Columns: {list(final_data.columns)}") 