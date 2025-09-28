#!/usr/bin/env python3
"""
Example Script for Stock Market Analysis and Forecasting

This script demonstrates a complete workflow for stock market analysis:
1. Data collection
2. Data preprocessing
3. Model training (ARIMA, SARIMA, Prophet, LSTM)
4. Model evaluation and comparison
5. Visualization

Run this script to see the project in action!
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.append('src')

# Import our modules
from data_collection.stock_data_collector import StockDataCollector
from data_collection.data_preprocessor import StockDataPreprocessor
from models.time_series_models import TimeSeriesModels
from models.lstm_model import LSTMModel
from visualization.stock_visualizer import StockVisualizer

def main():
    """Main function to demonstrate the complete workflow."""
    
    print("=" * 60)
    print("📈 Stock Market Analysis & Forecasting - Example")
    print("=" * 60)
    
    # Step 1: Initialize components
    print("\n🔧 Initializing components...")
    collector = StockDataCollector()
    preprocessor = StockDataPreprocessor()
    ts_models = TimeSeriesModels()
    visualizer = StockVisualizer()
    
    # Step 2: Collect data
    print("\n📥 Collecting stock data...")
    symbol = 'AAPL'  # Apple Inc.
    start_date = '2023-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Fetching data for {symbol} from {start_date} to {end_date}")
    
    data = collector.get_yahoo_finance_data(symbol, start_date=start_date, end_date=end_date)
    
    if data.empty:
        print("❌ No data collected. Please check your internet connection.")
        return
    
    print(f"✅ Collected {len(data)} data points for {symbol}")
    print(f"Data columns: {list(data.columns)}")
    
    # Step 3: Preprocess data
    print("\n🔧 Preprocessing data...")
    
    # Clean data
    cleaned_data = preprocessor.clean_data(data)
    print(f"✅ Data cleaned. Shape: {cleaned_data.shape}")
    
    # Add technical indicators
    data_with_indicators = preprocessor.add_technical_indicators(cleaned_data)
    print(f"✅ Technical indicators added. New columns: {len(data_with_indicators.columns)}")
    
    # Create features
    final_data = preprocessor.create_features(data_with_indicators)
    print(f"✅ Features created. Final shape: {final_data.shape}")
    
    # Step 4: Train time series models
    print("\n🤖 Training time series models...")
    
    close_prices = final_data['Close']
    
    # Train ARIMA
    print("Training ARIMA model...")
    arima_results = ts_models.fit_arima(close_prices, auto_optimize=True)
    
    # Train SARIMA
    print("Training SARIMA model...")
    sarima_results = ts_models.fit_sarima(close_prices, auto_optimize=True)
    
    # Train Prophet (if available)
    print("Training Prophet model...")
    prophet_results = ts_models.fit_prophet(final_data)
    
    # Step 5: Train LSTM model
    print("\n🧠 Training LSTM model...")
    
    # Initialize LSTM
    lstm_model = LSTMModel(
        sequence_length=30,
        n_features=5,
        n_lstm_layers=2,
        lstm_units=50,
        dense_units=25,
        dropout_rate=0.2
    )
    
    # Prepare sequences
    feature_columns = ['Close', 'Volume', 'RSI', 'MACD', 'Price_Change']
    available_features = [col for col in feature_columns if col in final_data.columns]
    
    if len(available_features) < 2:
        available_features = ['Close']
    
    X, y = lstm_model.prepare_sequences(final_data, target_column='Close', 
                                       feature_columns=available_features)
    
    if len(X) >= 100:
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        training_results = lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        
        # Evaluate model
        lstm_metrics = lstm_model.evaluate(X_test, y_test)
        lstm_predictions = lstm_model.predict(X_test)
        
        print("✅ LSTM model trained successfully!")
    else:
        print("⚠️ Insufficient data for LSTM training")
        lstm_metrics = {}
        lstm_predictions = None
    
    # Step 6: Model comparison
    print("\n📊 Comparing models...")
    
    # Collect predictions for comparison
    predictions = {}
    
    if arima_results and 'fitted_values' in arima_results:
        predictions['ARIMA'] = arima_results['fitted_values']
    
    if sarima_results and 'fitted_values' in sarima_results:
        predictions['SARIMA'] = sarima_results['fitted_values']
    
    if prophet_results and 'fitted_values' in prophet_results:
        prophet_fitted = prophet_results['fitted_values']
        predictions['Prophet'] = pd.Series(prophet_fitted, index=close_prices.index[:len(prophet_fitted)])
    
    if lstm_predictions is not None:
        test_indices = close_prices.index[-len(lstm_predictions):]
        predictions['LSTM'] = pd.Series(lstm_predictions, index=test_indices)
    
    # Compare models
    if predictions:
        comparison = ts_models.compare_models(close_prices, predictions)
        print("\n🏆 Model Comparison Results:")
        print(comparison)
        
        # Find best model
        if not comparison.empty and 'rmse' in comparison.columns:
            best_model = comparison['rmse'].idxmin()
            best_rmse = comparison.loc[best_model, 'rmse']
            print(f"\n🏆 Best performing model: {best_model} (RMSE: {best_rmse:.4f})")
    
    # Step 7: Create visualizations
    print("\n📈 Creating visualizations...")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Price chart
    price_chart = visualizer.plot_price_chart(final_data, symbol)
    price_file = visualizer.save_plot(price_chart, f"{symbol}_price_chart", 'html')
    print(f"📊 Price chart saved: {price_file}")
    
    # Technical indicators
    tech_chart = visualizer.plot_technical_indicators(final_data, symbol)
    tech_file = visualizer.save_plot(tech_chart, f"{symbol}_technical_indicators", 'html')
    print(f"📊 Technical indicators saved: {tech_file}")
    
    # Dashboard
    dashboard = visualizer.create_dashboard(final_data, symbol)
    dashboard_file = visualizer.save_plot(dashboard, f"{symbol}_dashboard", 'html')
    print(f"📊 Dashboard saved: {dashboard_file}")
    
    # Forecast comparison (if we have forecasts)
    if predictions:
        forecast_chart = visualizer.plot_forecast_comparison(close_prices, predictions, f"{symbol} Forecast Comparison")
        forecast_file = visualizer.save_plot(forecast_chart, f"{symbol}_forecast_comparison", 'html')
        print(f"📊 Forecast comparison saved: {forecast_file}")
    
    # Step 8: Save results
    print("\n💾 Saving results...")
    
    # Save processed data
    data_file = f"results/{symbol}_processed_data.csv"
    final_data.to_csv(data_file, index=False)
    print(f"📄 Processed data saved: {data_file}")
    
    # Save model comparison
    if not comparison.empty:
        comparison_file = f"results/{symbol}_model_comparison.csv"
        comparison.to_csv(comparison_file)
        print(f"📄 Model comparison saved: {comparison_file}")
    
    # Step 9: Summary
    print("\n" + "=" * 60)
    print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print(f"\n📊 Summary for {symbol}:")
    print(f"   • Data points: {len(final_data)}")
    print(f"   • Date range: {final_data['date'].min()} to {final_data['date'].max()}")
    print(f"   • Features: {len(final_data.columns)}")
    print(f"   • Models trained: {len(predictions)}")
    
    if not comparison.empty:
        print(f"   • Best model: {best_model}")
        print(f"   • Best RMSE: {best_rmse:.4f}")
    
    print(f"\n📁 Results saved in 'results' folder:")
    print(f"   • {symbol}_processed_data.csv")
    print(f"   • {symbol}_model_comparison.csv")
    print(f"   • {symbol}_price_chart.html")
    print(f"   • {symbol}_technical_indicators.html")
    print(f"   • {symbol}_dashboard.html")
    if predictions:
        print(f"   • {symbol}_forecast_comparison.html")
    
    print("\n🚀 Next steps:")
    print("   • Open the HTML files in your browser to view interactive charts")
    print("   • Run the Streamlit app: streamlit run app.py")
    print("   • Explore the Jupyter notebooks in the 'notebooks' folder")
    print("   • Try different stocks or time periods")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 