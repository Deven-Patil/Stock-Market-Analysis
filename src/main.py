"""
Main Application for Stock Market Analysis and Forecasting

This is the main entry point for the stock market analysis project.
It orchestrates the entire pipeline from data collection to model evaluation.
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Import our modules
from data_collection.stock_data_collector import StockDataCollector
from data_collection.data_preprocessor import StockDataPreprocessor
from models.time_series_models import TimeSeriesModels
# from models.lstm_model import LSTMModel
from visualization.stock_visualizer import StockVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StockMarketAnalyzer:
    """
    Main class to orchestrate the stock market analysis pipeline.
    """
    
    def __init__(self, 
                 symbols: List[str] = ['AAPL', 'MSFT', 'GOOGL'],
                 start_date: str = '2023-01-01',
                 end_date: str = None,
                 data_source: str = 'yahoo'):
        """
        Initialize the StockMarketAnalyzer.
        
        Args:
            symbols (List[str]): List of stock symbols to analyze
            start_date (str): Start date for data collection
            end_date (str): End date for data collection
            data_source (str): Data source ('yahoo' or 'alpha_vantage')
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data_source = data_source
        
        # Initialize components
        self.data_collector = StockDataCollector()
        self.preprocessor = StockDataPreprocessor()
        self.ts_models = TimeSeriesModels()
        self.visualizer = StockVisualizer()
        
        # Data storage
        self.raw_data = {}
        self.processed_data = {}
        self.models = {}
        self.results = {}
        
        logger.info(f"StockMarketAnalyzer initialized for symbols: {symbols}")
    
    def collect_data(self) -> Dict[str, pd.DataFrame]:
        """
        Collect stock data for all symbols.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of raw data for each symbol
        """
        logger.info("Starting data collection")
        
        try:
            # Collect data for all symbols
            self.raw_data = self.data_collector.get_multiple_stocks(
                symbols=self.symbols,
                source=self.data_source,
                start_date=self.start_date,
                end_date=self.end_date,
                period='1y' if self.data_source == 'yahoo' else None
            )
            
            # Save raw data
            for symbol, data in self.raw_data.items():
                if not data.empty:
                    filepath = self.data_collector.save_data(data, symbol, 'csv')
                    logger.info(f"Raw data saved for {symbol}: {filepath}")
            
            logger.info(f"Data collection completed. Collected data for {len(self.raw_data)} symbols")
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Error in data collection: {str(e)}")
            return {}
    
    def preprocess_data(self) -> Dict[str, pd.DataFrame]:
        """
        Preprocess all collected data.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of processed data for each symbol
        """
        logger.info("Starting data preprocessing")
        
        self.processed_data = {}
        
        for symbol, data in self.raw_data.items():
            if data.empty:
                continue
            
            logger.info(f"Preprocessing data for {symbol}")
            
            try:
                # Clean data
                cleaned_data = self.preprocessor.clean_data(data)
                
                # Add technical indicators
                data_with_indicators = self.preprocessor.add_technical_indicators(cleaned_data)
                
                # Create additional features
                final_data = self.preprocessor.create_features(data_with_indicators)
                
                # Store processed data
                self.processed_data[symbol] = final_data
                
                # Save processed data
                filepath = self.data_collector.save_data(final_data, f"{symbol}_processed", 'csv')
                logger.info(f"Processed data saved for {symbol}: {filepath}")
                
            except Exception as e:
                logger.error(f"Error preprocessing data for {symbol}: {str(e)}")
                continue
        
        logger.info(f"Data preprocessing completed for {len(self.processed_data)} symbols")
        return self.processed_data
    
    def train_time_series_models(self, symbol: str) -> Dict[str, Dict]:
        """
        Train time series models for a specific symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict[str, Dict]: Results for each model
        """
        if symbol not in self.processed_data:
            logger.error(f"No processed data available for {symbol}")
            return {}
        
        data = self.processed_data[symbol]
        close_prices = data['Close']
        
        logger.info(f"Training time series models for {symbol}")
        
        results = {}
        
        try:
            # Train ARIMA model
            arima_results = self.ts_models.fit_arima(close_prices, auto_optimize=True)
            if arima_results:
                results['arima'] = arima_results
            
            # Train SARIMA model
            sarima_results = self.ts_models.fit_sarima(close_prices, auto_optimize=True)
            if sarima_results:
                results['sarima'] = sarima_results
            
            # Train Prophet model
            prophet_results = self.ts_models.fit_prophet(data)
            if prophet_results:
                results['prophet'] = prophet_results
            
            self.models[symbol] = results
            logger.info(f"Time series models trained for {symbol}")
            
        except Exception as e:
            logger.error(f"Error training time series models for {symbol}: {str(e)}")
        
        return results
    
    def train_lstm_model(self, symbol: str) -> Dict:
        """
        Train LSTM model for a specific symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict: LSTM model results
        """
        # Lazy import to avoid TensorFlow requirement if not available
        try:
            from models.lstm_model import LSTMModel  # type: ignore
        except Exception as import_error:
            logger.warning("TensorFlow / LSTM dependencies not available. Skipping LSTM training. Error: %s", import_error)
            return {}
        
        if symbol not in self.processed_data:
            logger.error(f"No processed data available for {symbol}")
            return {}
        
        data = self.processed_data[symbol]
        
        logger.info(f"Training LSTM model for {symbol}")
        
        try:
            # Initialize LSTM model
            lstm_model = LSTMModel(
                sequence_length=60,
                n_features=5,  # Close, Volume, RSI, MACD, Price_Change
                n_lstm_layers=2,
                lstm_units=50,
                dense_units=25,
                dropout_rate=0.2
            )
            
            # Select features for LSTM
            feature_columns = ['Close', 'Volume', 'RSI', 'MACD', 'Price_Change']
            available_features = [col for col in feature_columns if col in data.columns]
            
            if len(available_features) < 2:
                available_features = ['Close']  # Fallback to just close price
            
            # Prepare sequences
            X, y = lstm_model.prepare_sequences(data, target_column='Close', 
                                               feature_columns=available_features)
            
            if len(X) < 100:
                logger.warning(f"Insufficient data for LSTM training for {symbol}")
                return {}
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train model
            training_results = lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            
            # Evaluate model
            evaluation_metrics = lstm_model.evaluate(X_test, y_test)
            
            # Make predictions
            predictions = lstm_model.predict(X_test)
            
            results = {
                'model': lstm_model,
                'training_results': training_results,
                'evaluation_metrics': evaluation_metrics,
                'predictions': predictions,
                'actual': y_test
            }
            
            # Store in models
            if symbol not in self.models:
                self.models[symbol] = {}
            self.models[symbol]['lstm'] = results
            
            logger.info(f"LSTM model trained for {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"Error training LSTM model for {symbol}: {str(e)}")
            return {}
    
    def create_visualizations(self, symbol: str) -> Dict[str, str]:
        """
        Create visualizations for a specific symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict[str, str]: Dictionary of saved visualization file paths
        """
        if symbol not in self.processed_data:
            logger.error(f"No processed data available for {symbol}")
            return {}
        
        data = self.processed_data[symbol]
        
        logger.info(f"Creating visualizations for {symbol}")
        
        saved_files = {}
        
        try:
            # Price chart
            price_chart = self.visualizer.plot_price_chart(data, symbol)
            price_file = self.visualizer.save_plot(price_chart, f"{symbol}_price_chart", 'html')
            if price_file:
                saved_files['price_chart'] = price_file
            
            # Technical indicators
            tech_chart = self.visualizer.plot_technical_indicators(data, symbol)
            tech_file = self.visualizer.save_plot(tech_chart, f"{symbol}_technical_indicators", 'html')
            if tech_file:
                saved_files['technical_indicators'] = tech_file
            
            # Dashboard
            dashboard = self.visualizer.create_dashboard(data, symbol)
            dashboard_file = self.visualizer.save_plot(dashboard, f"{symbol}_dashboard", 'html')
            if dashboard_file:
                saved_files['dashboard'] = dashboard_file
            
            # Returns distribution
            if 'Price_Change' in data.columns:
                returns_chart = self.visualizer.plot_returns_distribution(data)
                returns_file = self.visualizer.save_plot(returns_chart, f"{symbol}_returns_distribution", 'html')
                if returns_file:
                    saved_files['returns_distribution'] = returns_file
            
            logger.info(f"Visualizations created for {symbol}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations for {symbol}: {str(e)}")
        
        return saved_files
    
    def compare_models(self, symbol: str) -> pd.DataFrame:
        """
        Compare all trained models for a specific symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            pd.DataFrame: Model comparison results
        """
        if symbol not in self.models:
            logger.error(f"No models available for {symbol}")
            return pd.DataFrame()
        
        logger.info(f"Comparing models for {symbol}")
        
        try:
            # Get actual values
            data = self.processed_data[symbol]
            actual = data['Close']
            
            predictions = {}
            
            # Collect predictions from all models
            for model_name, model_results in self.models[symbol].items():
                if model_name == 'lstm':
                    # LSTM predictions are already available
                    if 'predictions' in model_results and 'actual' in model_results:
                        # Use test set predictions
                        test_indices = data.index[-len(model_results['predictions']):]
                        predictions[f'LSTM'] = pd.Series(model_results['predictions'], index=test_indices)
                
                elif model_name in ['arima', 'sarima']:
                    # Time series model fitted values
                    if 'fitted_values' in model_results:
                        predictions[f'{model_name.upper()}'] = model_results['fitted_values']
                
                elif model_name == 'prophet':
                    # Prophet fitted values
                    if 'fitted_values' in model_results:
                        predictions['Prophet'] = pd.Series(model_results['fitted_values'], index=data.index[:len(model_results['fitted_values'])])
            
            # Compare models
            if predictions:
                comparison = self.ts_models.compare_models(actual, predictions)
                
                # Save comparison results
                comparison_file = os.path.join('results', f'{symbol}_model_comparison.csv')
                comparison.to_csv(comparison_file)
                logger.info(f"Model comparison saved: {comparison_file}")
                
                return comparison
            
        except Exception as e:
            logger.error(f"Error comparing models for {symbol}: {str(e)}")
        
        return pd.DataFrame()
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline for all symbols.
        
        Returns:
            Dict[str, Any]: Complete analysis results
        """
        logger.info("Starting complete stock market analysis")
        
        results = {
            'symbols': self.symbols,
            'data_collection': {},
            'preprocessing': {},
            'models': {},
            'visualizations': {},
            'comparisons': {},
            'summary': {}
        }
        
        try:
            # Step 1: Collect data
            logger.info("Step 1: Data Collection")
            raw_data = self.collect_data()
            results['data_collection'] = {
                'collected_symbols': list(raw_data.keys()),
                'data_points': {symbol: len(data) for symbol, data in raw_data.items()}
            }
            
            # Step 2: Preprocess data
            logger.info("Step 2: Data Preprocessing")
            processed_data = self.preprocess_data()
            results['preprocessing'] = {
                'processed_symbols': list(processed_data.keys()),
                'features_added': len(processed_data.get(list(processed_data.keys())[0], pd.DataFrame()).columns) if processed_data else 0
            }
            
            # Step 3: Train models and create visualizations for each symbol
            for symbol in processed_data.keys():
                logger.info(f"Processing {symbol}")
                
                # Train time series models
                ts_results = self.train_time_series_models(symbol)
                results['models'][symbol] = {'time_series': ts_results}
                
                # Train LSTM model
                lstm_results = self.train_lstm_model(symbol)
                if lstm_results:
                    results['models'][symbol]['lstm'] = lstm_results
                
                # Create visualizations
                viz_files = self.create_visualizations(symbol)
                results['visualizations'][symbol] = viz_files
                
                # Compare models
                comparison = self.compare_models(symbol)
                results['comparisons'][symbol] = comparison.to_dict() if not comparison.empty else {}
            
            # Step 4: Generate summary
            logger.info("Step 4: Generating Summary")
            results['summary'] = self._generate_summary(results)
            
            # Save complete results
            self._save_results(results)
            
            logger.info("Complete analysis finished successfully")
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of the analysis results.
        
        Args:
            results (Dict[str, Any]): Analysis results
            
        Returns:
            Dict[str, Any]: Summary
        """
        summary = {
            'total_symbols': len(results['symbols']),
            'successful_symbols': len(results['preprocessing'].get('processed_symbols', [])),
            'models_trained': {},
            'best_models': {},
            'total_visualizations': sum(len(viz) for viz in results['visualizations'].values())
        }
        
        # Count models trained
        for symbol, models in results['models'].items():
            summary['models_trained'][symbol] = len(models)
        
        # Find best models
        for symbol, comparison in results['comparisons'].items():
            if comparison and 'rmse' in comparison:
                best_model = comparison['rmse'].idxmin()
                summary['best_models'][symbol] = best_model
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]) -> str:
        """
        Save analysis results to file.
        
        Args:
            results (Dict[str, Any]): Analysis results
            
        Returns:
            str: Path to saved results
        """
        import json
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join('results', f'analysis_results_{timestamp}.json')
        
        # Convert DataFrames to dict for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, pd.DataFrame):
                        serializable_results[key][sub_key] = sub_value.to_dict()
                    else:
                        serializable_results[key][sub_key] = sub_value
            else:
                serializable_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        return results_file

def main():
    """
    Main function to run the stock market analysis.
    """
    print("=" * 60)
    print("Stock Market Analysis and Forecasting Project")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = StockMarketAnalyzer(
        symbols=['AAPL', 'MSFT', 'GOOGL'],  # Popular tech stocks
        start_date='2023-01-01',
        data_source='yahoo'
    )
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    
    summary = results.get('summary', {})
    print(f"Total symbols analyzed: {summary.get('total_symbols', 0)}")
    print(f"Successful symbols: {summary.get('successful_symbols', 0)}")
    print(f"Total visualizations created: {summary.get('total_visualizations', 0)}")
    
    print("\nBest performing models:")
    for symbol, best_model in summary.get('best_models', {}).items():
        print(f"  {symbol}: {best_model}")
    
    print("\nAnalysis completed! Check the 'results' folder for outputs.")
    print("=" * 60)

if __name__ == "__main__":
    main() 