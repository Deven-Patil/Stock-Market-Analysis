"""
Time Series Models Module

This module implements various time series forecasting models including
ARIMA, SARIMA, and Facebook Prophet for stock price prediction.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Time series models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available. Install with: pip install prophet")

# Machine learning
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesModels:
    """
    A class to implement various time series forecasting models.
    """
    
    def __init__(self):
        """Initialize the TimeSeriesModels."""
        self.models = {}
        self.results = {}
        self.scaler = MinMaxScaler()
    
    def check_stationarity(self, 
                          data: pd.Series, 
                          method: str = 'adf') -> Dict[str, Any]:
        """
        Check if a time series is stationary.
        
        Args:
            data (pd.Series): Time series data
            method (str): Test method ('adf' for Augmented Dickey-Fuller, 'kpss' for KPSS)
            
        Returns:
            Dict[str, Any]: Test results
        """
        logger.info(f"Checking stationarity using {method.upper()} test")
        
        if method.lower() == 'adf':
            result = adfuller(data.dropna())
            is_stationary = result[1] < 0.05
            
            return {
                'test_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': is_stationary,
                'method': 'Augmented Dickey-Fuller'
            }
        
        elif method.lower() == 'kpss':
            result = kpss(data.dropna())
            is_stationary = result[1] > 0.05
            
            return {
                'test_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[3],
                'is_stationary': is_stationary,
                'method': 'KPSS'
            }
        
        else:
            raise ValueError("Method must be 'adf' or 'kpss'")
    
    def make_stationary(self, 
                       data: pd.Series, 
                       method: str = 'diff') -> Tuple[pd.Series, int]:
        """
        Make time series stationary using differencing or other methods.
        
        Args:
            data (pd.Series): Time series data
            method (str): Method to make stationary ('diff', 'log_diff')
            
        Returns:
            Tuple[pd.Series, int]: Stationary series and number of differences
        """
        logger.info(f"Making series stationary using {method}")
        
        original_data = data.copy()
        n_diff = 0
        
        if method == 'diff':
            # Simple differencing
            while n_diff < 3:  # Maximum 3 differences
                test_result = self.check_stationarity(data)
                if test_result['is_stationary']:
                    break
                
                data = data.diff().dropna()
                n_diff += 1
                logger.info(f"Applied {n_diff} difference(s)")
        
        elif method == 'log_diff':
            # Log transformation followed by differencing
            if (data > 0).all():
                data = np.log(data)
                data = data.diff().dropna()
                n_diff = 1
            else:
                logger.warning("Cannot apply log transformation to non-positive data")
                return original_data, 0
        
        logger.info(f"Series made stationary with {n_diff} difference(s)")
        return data, n_diff
    
    def find_optimal_arima_params(self, 
                                 data: pd.Series, 
                                 max_p: int = 3, 
                                 max_d: int = 2, 
                                 max_q: int = 3) -> Tuple[int, int, int]:
        """
        Find optimal ARIMA parameters using grid search.
        
        Args:
            data (pd.Series): Time series data
            max_p (int): Maximum AR order
            max_d (int): Maximum differencing order
            max_q (int): Maximum MA order
            
        Returns:
            Tuple[int, int, int]: Optimal (p, d, q) parameters
        """
        logger.info("Finding optimal ARIMA parameters")
        
        best_aic = np.inf
        best_params = (0, 0, 0)
        
        # Make data stationary first
        stationary_data, d = self.make_stationary(data)
        
        # Grid search for p and q
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(stationary_data, order=(p, 0, q))
                    fitted_model = model.fit()
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_params = (p, d, q)
                        
                except:
                    continue
        
        logger.info(f"Optimal ARIMA parameters: {best_params} (AIC: {best_aic:.2f})")
        return best_params
    
    def fit_arima(self, 
                 data: pd.Series, 
                 order: Tuple[int, int, int] = None,
                 auto_optimize: bool = True) -> Dict[str, Any]:
        """
        Fit ARIMA model to the data.
        
        Args:
            data (pd.Series): Time series data
            order (Tuple[int, int, int]): ARIMA order (p, d, q)
            auto_optimize (bool): Whether to automatically find optimal parameters
            
        Returns:
            Dict[str, Any]: Model results
        """
        logger.info("Fitting ARIMA model")
        
        if auto_optimize and order is None:
            order = self.find_optimal_arima_params(data)
        
        if order is None:
            order = (1, 1, 1)  # Default order
        
        try:
            model = ARIMA(data, order=order)
            fitted_model = model.fit()
            
            # Store model
            self.models['arima'] = fitted_model
            
            # Make predictions
            forecast = fitted_model.forecast(steps=30)  # 30 days ahead
            fitted_values = fitted_model.fittedvalues
            
            results = {
                'model': fitted_model,
                'order': order,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'forecast': forecast,
                'fitted_values': fitted_values,
                'residuals': fitted_model.resid
            }
            
            self.results['arima'] = results
            logger.info(f"ARIMA model fitted successfully. AIC: {fitted_model.aic:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {str(e)}")
            return {}
    
    def fit_sarima(self, 
                  data: pd.Series, 
                  order: Tuple[int, int, int] = None,
                  seasonal_order: Tuple[int, int, int, int] = None,
                  auto_optimize: bool = True) -> Dict[str, Any]:
        """
        Fit SARIMA model to the data.
        
        Args:
            data (pd.Series): Time series data
            order (Tuple[int, int, int]): SARIMA order (p, d, q)
            seasonal_order (Tuple[int, int, int, int]): Seasonal order (P, D, Q, s)
            auto_optimize (bool): Whether to automatically find optimal parameters
            
        Returns:
            Dict[str, Any]: Model results
        """
        logger.info("Fitting SARIMA model")
        
        if auto_optimize and order is None:
            order = self.find_optimal_arima_params(data)
        
        if order is None:
            order = (1, 1, 1)
        
        if seasonal_order is None:
            seasonal_order = (1, 1, 1, 12)  # Monthly seasonality
        
        try:
            model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit(disp=False)
            
            # Store model
            self.models['sarima'] = fitted_model
            
            # Make predictions
            forecast = fitted_model.forecast(steps=30)
            fitted_values = fitted_model.fittedvalues
            
            results = {
                'model': fitted_model,
                'order': order,
                'seasonal_order': seasonal_order,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'forecast': forecast,
                'fitted_values': fitted_values,
                'residuals': fitted_model.resid
            }
            
            self.results['sarima'] = results
            logger.info(f"SARIMA model fitted successfully. AIC: {fitted_model.aic:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error fitting SARIMA model: {str(e)}")
            return {}
    
    def fit_prophet(self, 
                   data: pd.DataFrame, 
                   date_col: str = 'date',
                   value_col: str = 'Close',
                   **kwargs) -> Dict[str, Any]:
        """
        Fit Facebook Prophet model to the data.
        
        Args:
            data (pd.DataFrame): Time series data
            date_col (str): Date column name
            value_col (str): Value column name
            **kwargs: Additional Prophet parameters
            
        Returns:
            Dict[str, Any]: Model results
        """
        if not PROPHET_AVAILABLE:
            logger.error("Prophet is not available. Please install it first.")
            return {}
        
        logger.info("Fitting Prophet model")
        
        # Prepare data for Prophet
        prophet_data = data[[date_col, value_col]].copy()
        prophet_data.columns = ['ds', 'y']
        
        # Initialize and fit model
        model = Prophet(**kwargs)
        model.fit(prophet_data)
        
        # Store model
        self.models['prophet'] = model
        
        # Make predictions
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        results = {
            'model': model,
            'forecast': forecast,
            'fitted_values': forecast['yhat'][:len(prophet_data)],
            'residuals': prophet_data['y'] - forecast['yhat'][:len(prophet_data)]
        }
        
        self.results['prophet'] = results
        logger.info("Prophet model fitted successfully")
        
        return results
    
    def evaluate_model(self, 
                      actual: pd.Series, 
                      predicted: pd.Series,
                      model_name: str = 'model') -> Dict[str, float]:
        """
        Evaluate model performance using various metrics.
        
        Args:
            actual (pd.Series): Actual values
            predicted (pd.Series): Predicted values
            model_name (str): Name of the model
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        logger.info(f"Evaluating {model_name}")
        
        # Remove NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            logger.warning("No valid data for evaluation")
            return {}
        
        metrics = {
            'mse': mean_squared_error(actual_clean, predicted_clean),
            'rmse': np.sqrt(mean_squared_error(actual_clean, predicted_clean)),
            'mae': mean_absolute_error(actual_clean, predicted_clean),
            'mape': np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100,
            'r2': r2_score(actual_clean, predicted_clean)
        }
        
        logger.info(f"{model_name} evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
        
        return metrics
    
    def compare_models(self, 
                      actual: pd.Series, 
                      predictions: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Compare multiple models using various metrics.
        
        Args:
            actual (pd.Series): Actual values
            predictions (Dict[str, pd.Series]): Dictionary of predictions
            
        Returns:
            pd.DataFrame: Comparison results
        """
        logger.info("Comparing models")
        
        comparison_results = []
        
        for model_name, predicted in predictions.items():
            metrics = self.evaluate_model(actual, predicted, model_name)
            if metrics:
                metrics['model'] = model_name
                comparison_results.append(metrics)
        
        if comparison_results:
            comparison_df = pd.DataFrame(comparison_results)
            comparison_df = comparison_df.set_index('model')
            
            logger.info("Model comparison completed")
            return comparison_df
        
        return pd.DataFrame()
    
    def plot_forecast(self, 
                     actual: pd.Series, 
                     forecast: pd.Series,
                     title: str = 'Forecast vs Actual') -> None:
        """
        Plot forecast vs actual values.
        
        Args:
            actual (pd.Series): Actual values
            forecast (pd.Series): Forecasted values
            title (str): Plot title
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.plot(actual.index, actual.values, label='Actual', color='blue')
        plt.plot(forecast.index, forecast.values, label='Forecast', color='red', linestyle='--')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_forecast_confidence_intervals(self, 
                                        model_name: str = 'prophet',
                                        periods: int = 30) -> pd.DataFrame:
        """
        Get forecast confidence intervals.
        
        Args:
            model_name (str): Name of the model
            periods (int): Number of periods to forecast
            
        Returns:
            pd.DataFrame: Forecast with confidence intervals
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return pd.DataFrame()
        
        if model_name == 'prophet':
            model = self.models[model_name]
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
        else:
            logger.warning(f"Confidence intervals not implemented for {model_name}")
            return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_data = pd.Series(np.random.randn(100).cumsum(), index=dates)
    
    # Initialize models
    ts_models = TimeSeriesModels()
    
    # Fit ARIMA model
    arima_results = ts_models.fit_arima(sample_data, auto_optimize=True)
    
    # Fit SARIMA model
    sarima_results = ts_models.fit_sarima(sample_data, auto_optimize=True)
    
    print("Time series models completed!") 