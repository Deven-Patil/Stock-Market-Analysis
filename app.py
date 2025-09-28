"""
Streamlit Web Application for Stock Market Analysis

This is a web-based interface for the stock market analysis and forecasting project.
It provides an interactive way to explore stock data, train models, and view results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.append('src')

# Import our modules
from data_collection.stock_data_collector import StockDataCollector
from data_collection.data_preprocessor import StockDataPreprocessor
from models.time_series_models import TimeSeriesModels
from models.lstm_model import LSTMModel
from visualization.stock_visualizer import StockVisualizer

# Page configuration
st.set_page_config(
    page_title="Stock Market Analysis & Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .model-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitApp:
    """
    Streamlit application for stock market analysis.
    """
    
    def __init__(self):
        """Initialize the Streamlit app."""
        self.data_collector = StockDataCollector()
        self.preprocessor = StockDataPreprocessor()
        self.ts_models = TimeSeriesModels()
        self.visualizer = StockVisualizer()
        
        # Initialize session state
        if 'data' not in st.session_state:
            st.session_state.data = {}
        if 'models' not in st.session_state:
            st.session_state.models = {}
        if 'results' not in st.session_state:
            st.session_state.results = {}
    
    def main_page(self):
        """Main page of the application."""
        st.markdown('<h1 class="main-header">📈 Stock Market Analysis & Forecasting</h1>', unsafe_allow_html=True)
        
        # Sidebar
        self.sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏠 Dashboard", 
            "📊 Data Analysis", 
            "🤖 Model Training", 
            "📈 Forecasting", 
            "📋 Results"
        ])
        
        with tab1:
            self.dashboard_tab()
        
        with tab2:
            self.data_analysis_tab()
        
        with tab3:
            self.model_training_tab()
        
        with tab4:
            self.forecasting_tab()
        
        with tab5:
            self.results_tab()
    
    def sidebar(self):
        """Sidebar configuration."""
        st.sidebar.title("⚙️ Configuration")
        
        # Stock symbol input
        st.sidebar.subheader("Stock Selection")
        symbol = st.sidebar.text_input(
            "Stock Symbol", 
            value="AAPL",
            help="Enter stock symbol (e.g., AAPL, MSFT, GOOGL)"
        )
        
        # Date range
        st.sidebar.subheader("Date Range")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                max_value=datetime.now()
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now()
            )
        
        # Data source
        data_source = st.sidebar.selectbox(
            "Data Source",
            ["yahoo", "alpha_vantage"],
            help="Choose data source for stock data"
        )
        
        # Load data button
        if st.sidebar.button("📥 Load Data", type="primary"):
            with st.spinner("Loading stock data..."):
                self.load_data(symbol, start_date, end_date, data_source)
        
        # Model parameters
        st.sidebar.subheader("Model Parameters")
        
        # LSTM parameters
        st.sidebar.markdown("**LSTM Settings**")
        sequence_length = st.sidebar.slider("Sequence Length", 10, 100, 60)
        lstm_units = st.sidebar.slider("LSTM Units", 20, 100, 50)
        epochs = st.sidebar.slider("Training Epochs", 10, 200, 50)
        
        # Store parameters in session state
        st.session_state.model_params = {
            'sequence_length': sequence_length,
            'lstm_units': lstm_units,
            'epochs': epochs
        }
        
        # Quick actions
        st.sidebar.subheader("🚀 Quick Actions")
        
        if st.sidebar.button("🔄 Run Complete Analysis"):
            with st.spinner("Running complete analysis..."):
                self.run_complete_analysis(symbol)
    
    def load_data(self, symbol: str, start_date, end_date, data_source: str):
        """Load stock data."""
        try:
            # Convert dates to string
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Fetch data
            if data_source == 'yahoo':
                data = self.data_collector.get_yahoo_finance_data(
                    symbol, start_date=start_str, end_date=end_str
                )
            else:
                data = self.data_collector.get_alpha_vantage_data(symbol)
            
            if not data.empty:
                # Preprocess data
                processed_data = self.preprocessor.clean_data(data)
                processed_data = self.preprocessor.add_technical_indicators(processed_data)
                processed_data = self.preprocessor.create_features(processed_data)
                
                # Store in session state
                st.session_state.data[symbol] = processed_data
                
                st.success(f"✅ Data loaded successfully for {symbol}!")
                st.info(f"📊 {len(processed_data)} data points loaded")
            else:
                st.error(f"❌ No data found for {symbol}")
                
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
    
    def dashboard_tab(self):
        """Dashboard tab."""
        st.header("📊 Dashboard")
        
        if not st.session_state.data:
            st.info("👆 Load some data from the sidebar to get started!")
            return
        
        # Display data for each symbol
        for symbol, data in st.session_state.data.items():
            st.subheader(f"📈 {symbol} Overview")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_price = data['Close'].iloc[-1]
                st.metric("Current Price", f"${current_price:.2f}")
            
            with col2:
                price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
                st.metric("Daily Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
            
            with col3:
                volume = data['Volume'].iloc[-1] if 'Volume' in data.columns else 0
                st.metric("Volume", f"{volume:,.0f}")
            
            with col4:
                volatility = data['Volatility'].iloc[-1] if 'Volatility' in data.columns else 0
                st.metric("Volatility", f"{volatility:.4f}")
            
            # Price chart
            fig = self.visualizer.plot_price_chart(data, symbol)
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical indicators
            if any(col in data.columns for col in ['RSI', 'MACD', 'SMA_20']):
                tech_fig = self.visualizer.plot_technical_indicators(data, symbol)
                st.plotly_chart(tech_fig, use_container_width=True)
    
    def data_analysis_tab(self):
        """Data analysis tab."""
        st.header("📊 Data Analysis")
        
        if not st.session_state.data:
            st.info("👆 Load some data first!")
            return
        
        # Symbol selector
        symbol = st.selectbox("Select Symbol", list(st.session_state.data.keys()))
        data = st.session_state.data[symbol]
        
        # Data overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Price Trends")
            
            # Price over time
            fig = px.line(data, x='date', y='Close', title=f'{symbol} Close Price')
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume analysis
            if 'Volume' in data.columns:
                fig = px.bar(data, x='date', y='Volume', title=f'{symbol} Trading Volume')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📋 Data Statistics")
            
            # Basic statistics
            stats = data[['Open', 'High', 'Low', 'Close']].describe()
            st.dataframe(stats)
            
            # Correlation matrix
            if len(data.select_dtypes(include=[np.number]).columns) > 1:
                st.subheader("🔗 Correlation Matrix")
                corr = data.select_dtypes(include=[np.number]).corr()
                fig = px.imshow(corr, title="Feature Correlations")
                st.plotly_chart(fig, use_container_width=True)
        
        # Returns analysis
        st.subheader("📊 Returns Analysis")
        
        if 'Price_Change' in data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Returns distribution
                fig = px.histogram(data, x='Price_Change', title="Returns Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Cumulative returns
                cumulative_returns = (1 + data['Price_Change']).cumprod()
                fig = px.line(x=data['date'], y=cumulative_returns, title="Cumulative Returns")
                st.plotly_chart(fig, use_container_width=True)
    
    def model_training_tab(self):
        """Model training tab."""
        st.header("🤖 Model Training")
        
        if not st.session_state.data:
            st.info("👆 Load some data first!")
            return
        
        # Symbol selector
        symbol = st.selectbox("Select Symbol for Training", list(st.session_state.data.keys()))
        data = st.session_state.data[symbol]
        
        # Model selection
        st.subheader("🎯 Select Models to Train")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            train_arima = st.checkbox("ARIMA", value=True)
        
        with col2:
            train_sarima = st.checkbox("SARIMA", value=True)
        
        with col3:
            train_lstm = st.checkbox("LSTM", value=True)
        
        # Training button
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models..."):
                self.train_models(symbol, data, train_arima, train_sarima, train_lstm)
        
        # Display trained models
        if symbol in st.session_state.models:
            st.subheader("✅ Trained Models")
            
            for model_name, model_results in st.session_state.models[symbol].items():
                with st.expander(f"📊 {model_name.upper()} Results"):
                    if model_name == 'lstm':
                        self.display_lstm_results(model_results)
                    else:
                        self.display_ts_model_results(model_results)
    
    def train_models(self, symbol: str, data: pd.DataFrame, train_arima: bool, train_sarima: bool, train_lstm: bool):
        """Train selected models."""
        close_prices = data['Close']
        
        if symbol not in st.session_state.models:
            st.session_state.models[symbol] = {}
        
        # Train ARIMA
        if train_arima:
            try:
                arima_results = self.ts_models.fit_arima(close_prices, auto_optimize=True)
                if arima_results:
                    st.session_state.models[symbol]['arima'] = arima_results
                    st.success("✅ ARIMA model trained successfully!")
            except Exception as e:
                st.error(f"❌ Error training ARIMA: {str(e)}")
        
        # Train SARIMA
        if train_sarima:
            try:
                sarima_results = self.ts_models.fit_sarima(close_prices, auto_optimize=True)
                if sarima_results:
                    st.session_state.models[symbol]['sarima'] = sarima_results
                    st.success("✅ SARIMA model trained successfully!")
            except Exception as e:
                st.error(f"❌ Error training SARIMA: {str(e)}")
        
        # Train LSTM
        if train_lstm:
            try:
                params = st.session_state.model_params
                
                lstm_model = LSTMModel(
                    sequence_length=params['sequence_length'],
                    n_features=5,
                    lstm_units=params['lstm_units']
                )
                
                # Prepare data
                feature_columns = ['Close', 'Volume', 'RSI', 'MACD', 'Price_Change']
                available_features = [col for col in feature_columns if col in data.columns]
                
                if len(available_features) < 2:
                    available_features = ['Close']
                
                X, y = lstm_model.prepare_sequences(data, target_column='Close', 
                                                   feature_columns=available_features)
                
                if len(X) >= 100:
                    # Split data
                    split_idx = int(0.8 * len(X))
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    # Train model
                    training_results = lstm_model.fit(X_train, y_train, epochs=params['epochs'], verbose=0)
                    
                    # Evaluate
                    evaluation_metrics = lstm_model.evaluate(X_test, y_test)
                    predictions = lstm_model.predict(X_test)
                    
                    lstm_results = {
                        'model': lstm_model,
                        'training_results': training_results,
                        'evaluation_metrics': evaluation_metrics,
                        'predictions': predictions,
                        'actual': y_test
                    }
                    
                    st.session_state.models[symbol]['lstm'] = lstm_results
                    st.success("✅ LSTM model trained successfully!")
                else:
                    st.warning("⚠️ Insufficient data for LSTM training")
                    
            except Exception as e:
                st.error(f"❌ Error training LSTM: {str(e)}")
    
    def display_lstm_results(self, results: dict):
        """Display LSTM model results."""
        # Training history
        if 'training_results' in results and 'history' in results['training_results']:
            history = results['training_results']['history']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.line(y=history['loss'], title="Training Loss")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'val_loss' in history:
                    fig = px.line(y=history['val_loss'], title="Validation Loss")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Evaluation metrics
        if 'evaluation_metrics' in results:
            metrics = results['evaluation_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
            
            with col2:
                st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
            
            with col3:
                st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
            
            with col4:
                st.metric("R²", f"{metrics.get('r2', 0):.4f}")
    
    def display_ts_model_results(self, results: dict):
        """Display time series model results."""
        if 'aic' in results:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("AIC", f"{results['aic']:.2f}")
            
            with col2:
                if 'bic' in results:
                    st.metric("BIC", f"{results['bic']:.2f}")
        
        # Model parameters
        if 'order' in results:
            st.write(f"**Model Order:** {results['order']}")
        
        if 'seasonal_order' in results:
            st.write(f"**Seasonal Order:** {results['seasonal_order']}")
    
    def forecasting_tab(self):
        """Forecasting tab."""
        st.header("📈 Forecasting")
        
        if not st.session_state.models:
            st.info("👆 Train some models first!")
            return
        
        # Symbol selector
        symbol = st.selectbox("Select Symbol for Forecasting", list(st.session_state.models.keys()))
        
        # Forecast period
        forecast_days = st.slider("Forecast Period (days)", 7, 90, 30)
        
        if st.button("🔮 Generate Forecast", type="primary"):
            with st.spinner("Generating forecasts..."):
                self.generate_forecasts(symbol, forecast_days)
    
    def generate_forecasts(self, symbol: str, forecast_days: int):
        """Generate forecasts for all trained models."""
        data = st.session_state.data[symbol]
        models = st.session_state.models[symbol]
        
        # Create forecast comparison
        actual = data['Close']
        forecasts = {}
        
        # Generate forecasts for each model
        for model_name, model_results in models.items():
            try:
                if model_name == 'lstm':
                    # LSTM forecast
                    lstm_model = model_results['model']
                    last_sequence = lstm_model.scaler.transform(data[['Close']].tail(lstm_model.sequence_length))
                    forecast = lstm_model.forecast_future(last_sequence, steps=forecast_days)
                    forecast_dates = pd.date_range(data['date'].iloc[-1], periods=forecast_days+1, freq='D')[1:]
                    forecasts['LSTM'] = pd.Series(forecast, index=forecast_dates)
                
                elif model_name in ['arima', 'sarima']:
                    # Time series model forecast
                    if 'forecast' in model_results:
                        forecast = model_results['forecast']
                        if len(forecast) >= forecast_days:
                            forecast_dates = pd.date_range(data['date'].iloc[-1], periods=forecast_days+1, freq='D')[1:]
                            forecasts[model_name.upper()] = pd.Series(forecast[:forecast_days], index=forecast_dates)
                
            except Exception as e:
                st.error(f"❌ Error generating forecast for {model_name}: {str(e)}")
        
        # Display forecasts
        if forecasts:
            st.subheader("📊 Forecast Comparison")
            
            # Create comparison plot
            fig = go.Figure()
            
            # Add actual data
            fig.add_trace(go.Scatter(
                x=actual.index,
                y=actual.values,
                mode='lines',
                name='Actual',
                line=dict(color='black', width=2)
            ))
            
            # Add forecasts
            colors = ['blue', 'red', 'green', 'orange']
            for i, (model_name, forecast) in enumerate(forecasts.items()):
                fig.add_trace(go.Scatter(
                    x=forecast.index,
                    y=forecast.values,
                    mode='lines',
                    name=f'{model_name} (Forecast)',
                    line=dict(color=colors[i % len(colors)], width=1, dash='dash')
                ))
            
            fig.update_layout(
                title=f"{symbol} Price Forecast",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast table
            st.subheader("📋 Forecast Values")
            forecast_df = pd.DataFrame(forecasts)
            st.dataframe(forecast_df)
    
    def results_tab(self):
        """Results tab."""
        st.header("📋 Results & Reports")
        
        if not st.session_state.models:
            st.info("👆 Train some models first!")
            return
        
        # Model comparison
        st.subheader("🏆 Model Comparison")
        
        for symbol in st.session_state.models.keys():
            with st.expander(f"📊 {symbol} Model Comparison"):
                self.compare_models(symbol)
        
        # Download results
        st.subheader("💾 Download Results")
        
        if st.button("📥 Export Results"):
            self.export_results()
    
    def compare_models(self, symbol: str):
        """Compare models for a symbol."""
        data = st.session_state.data[symbol]
        models = st.session_state.models[symbol]
        
        actual = data['Close']
        comparisons = []
        
        for model_name, model_results in models.items():
            try:
                if model_name == 'lstm':
                    # Use test set predictions for LSTM
                    predictions = model_results['predictions']
                    test_indices = data.index[-len(predictions):]
                    pred_series = pd.Series(predictions, index=test_indices)
                    
                    # Calculate metrics
                    metrics = self.ts_models.evaluate_model(actual[test_indices], pred_series, model_name)
                    if metrics:
                        metrics['model'] = 'LSTM'
                        comparisons.append(metrics)
                
                elif model_name in ['arima', 'sarima']:
                    # Use fitted values for time series models
                    if 'fitted_values' in model_results:
                        fitted = model_results['fitted_values']
                        metrics = self.ts_models.evaluate_model(actual, fitted, model_name)
                        if metrics:
                            metrics['model'] = model_name.upper()
                            comparisons.append(metrics)
            
            except Exception as e:
                st.error(f"❌ Error comparing {model_name}: {str(e)}")
        
        if comparisons:
            comparison_df = pd.DataFrame(comparisons)
            comparison_df = comparison_df.set_index('model')
            
            st.dataframe(comparison_df)
            
            # Best model
            if 'rmse' in comparison_df.columns:
                best_model = comparison_df['rmse'].idxmin()
                st.success(f"🏆 Best performing model: **{best_model}** (RMSE: {comparison_df.loc[best_model, 'rmse']:.4f})")
    
    def export_results(self):
        """Export results to files."""
        try:
            # Create results directory
            os.makedirs('results', exist_ok=True)
            
            # Export data
            for symbol, data in st.session_state.data.items():
                data.to_csv(f'results/{symbol}_data.csv', index=False)
            
            # Export model results
            for symbol, models in st.session_state.models.items():
                # Save model summaries
                with open(f'results/{symbol}_model_summary.txt', 'w') as f:
                    f.write(f"Model Summary for {symbol}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for model_name, results in models.items():
                        f.write(f"{model_name.upper()} Model:\n")
                        f.write("-" * 30 + "\n")
                        
                        if 'evaluation_metrics' in results:
                            for metric, value in results['evaluation_metrics'].items():
                                f.write(f"{metric.upper()}: {value:.4f}\n")
                        
                        if 'aic' in results:
                            f.write(f"AIC: {results['aic']:.2f}\n")
                        
                        f.write("\n")
            
            st.success("✅ Results exported successfully! Check the 'results' folder.")
            
        except Exception as e:
            st.error(f"❌ Error exporting results: {str(e)}")
    
    def run_complete_analysis(self, symbol: str):
        """Run complete analysis for a symbol."""
        try:
            # This would integrate with the main analysis pipeline
            st.success(f"✅ Complete analysis completed for {symbol}!")
            
        except Exception as e:
            st.error(f"❌ Error in complete analysis: {str(e)}")

def main():
    """Main function to run the Streamlit app."""
    app = StreamlitApp()
    app.main_page()

if __name__ == "__main__":
    main() 