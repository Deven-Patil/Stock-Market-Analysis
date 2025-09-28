"""
Stock Market Visualization Module

This module provides comprehensive visualization capabilities for stock market data
including price charts, technical indicators, and interactive dashboards.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Tuple
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StockVisualizer:
    """
    A class to create various visualizations for stock market data.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the StockVisualizer.
        
        Args:
            figsize (Tuple[int, int]): Default figure size for matplotlib plots
        """
        self.figsize = figsize
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
    
    def plot_price_chart(self, 
                        data: pd.DataFrame, 
                        symbol: str = '',
                        interactive: bool = True,
                        show_volume: bool = True) -> go.Figure:
        """
        Create a comprehensive price chart with candlesticks and volume.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            symbol (str): Stock symbol for title
            interactive (bool): Whether to create interactive plotly chart
            show_volume (bool): Whether to show volume bars
            
        Returns:
            go.Figure: Plotly figure object
        """
        if interactive:
            return self._create_interactive_price_chart(data, symbol, show_volume)
        else:
            return self._create_static_price_chart(data, symbol, show_volume)
    
    def _create_interactive_price_chart(self, 
                                      data: pd.DataFrame, 
                                      symbol: str,
                                      show_volume: bool) -> go.Figure:
        """Create interactive price chart using Plotly."""
        
        # Determine subplot layout
        if show_volume and 'Volume' in data.columns:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f'{symbol} Stock Price', 'Volume'),
                row_width=[0.7, 0.3]
            )
        else:
            fig = make_subplots(
                rows=1, cols=1,
                subplot_titles=(f'{symbol} Stock Price',)
            )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data['date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC',
                increasing_line_color=self.colors['success'],
                decreasing_line_color=self.colors['danger']
            ),
            row=1, col=1
        )
        
        # Add volume bars if requested
        if show_volume and 'Volume' in data.columns:
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(data['Close'], data['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=data['date'],
                    y=data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Stock Price Chart',
            yaxis_title='Price ($)',
            xaxis_rangeslider_visible=False,
            height=600,
            showlegend=True
        )
        
        if show_volume and 'Volume' in data.columns:
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def _create_static_price_chart(self, 
                                 data: pd.DataFrame, 
                                 symbol: str,
                                 show_volume: bool) -> plt.Figure:
        """Create static price chart using Matplotlib."""
        
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, 
                                gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot candlestick chart
        self._plot_candlesticks(axes[0], data)
        axes[0].set_title(f'{symbol} Stock Price')
        axes[0].set_ylabel('Price ($)')
        axes[0].grid(True, alpha=0.3)
        
        # Plot volume if requested
        if show_volume and 'Volume' in data.columns:
            self._plot_volume(axes[1], data)
            axes[1].set_ylabel('Volume')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_candlesticks(self, ax: plt.Axes, data: pd.DataFrame):
        """Plot candlestick chart using matplotlib."""
        
        # Calculate candlestick properties
        width = 0.6
        width2 = width * 0.8
        
        up = data[data.Close >= data.Open]
        down = data[data.Close < data.Open]
        
        # Up candlesticks
        ax.bar(up.index, up.Close - up.Open, width, bottom=up.Open, 
               color=self.colors['success'], alpha=0.7)
        ax.bar(up.index, up.High - up.Close, width2, bottom=up.Close, 
               color=self.colors['success'])
        ax.bar(up.index, up.Low - up.Open, width2, bottom=up.Open, 
               color=self.colors['success'])
        
        # Down candlesticks
        ax.bar(down.index, down.Close - down.Open, width, bottom=down.Open, 
               color=self.colors['danger'], alpha=0.7)
        ax.bar(down.index, down.High - down.Open, width2, bottom=down.Open, 
               color=self.colors['danger'])
        ax.bar(down.index, down.Low - down.Close, width2, bottom=down.Close, 
               color=self.colors['danger'])
        
        ax.set_xlim(-1, len(data) + 1)
    
    def _plot_volume(self, ax: plt.Axes, data: pd.DataFrame):
        """Plot volume bars."""
        
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(data['Close'], data['Open'])]
        
        ax.bar(range(len(data)), data['Volume'], color=colors, alpha=0.7)
        ax.set_xlim(-1, len(data) + 1)
    
    def plot_technical_indicators(self, 
                                data: pd.DataFrame, 
                                symbol: str = '',
                                indicators: List[str] = None) -> go.Figure:
        """
        Plot technical indicators.
        
        Args:
            data (pd.DataFrame): Stock data with technical indicators
            symbol (str): Stock symbol
            indicators (List[str]): List of indicators to plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        if indicators is None:
            indicators = ['SMA_20', 'EMA_12', 'RSI', 'MACD']
        
        # Create subplots
        n_indicators = len(indicators)
        fig = make_subplots(
            rows=n_indicators + 1, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[f'{symbol} Price'] + indicators
        )
        
        # Plot price
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color=self.colors['primary'])
            ),
            row=1, col=1
        )
        
        # Plot indicators
        for i, indicator in enumerate(indicators, 2):
            if indicator in data.columns:
                if 'SMA' in indicator or 'EMA' in indicator:
                    # Moving averages
                    fig.add_trace(
                        go.Scatter(
                            x=data['date'],
                            y=data[indicator],
                            mode='lines',
                            name=indicator,
                            line=dict(color=self.colors['secondary'])
                        ),
                        row=i, col=1
                    )
                
                elif indicator == 'RSI':
                    # RSI with overbought/oversold lines
                    fig.add_trace(
                        go.Scatter(
                            x=data['date'],
                            y=data[indicator],
                            mode='lines',
                            name='RSI',
                            line=dict(color=self.colors['primary'])
                        ),
                        row=i, col=1
                    )
                    
                    # Add overbought/oversold lines
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=i, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=i, col=1)
                
                elif indicator == 'MACD':
                    # MACD
                    fig.add_trace(
                        go.Scatter(
                            x=data['date'],
                            y=data['MACD'],
                            mode='lines',
                            name='MACD',
                            line=dict(color=self.colors['primary'])
                        ),
                        row=i, col=1
                    )
                    
                    if 'MACD_Signal' in data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=data['date'],
                                y=data['MACD_Signal'],
                                mode='lines',
                                name='MACD Signal',
                                line=dict(color=self.colors['secondary'])
                            ),
                            row=i, col=1
                        )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Technical Indicators',
            height=200 * (n_indicators + 1),
            showlegend=True
        )
        
        return fig
    
    def plot_correlation_matrix(self, 
                              data: pd.DataFrame, 
                              columns: Optional[List[str]] = None) -> plt.Figure:
        """
        Plot correlation matrix heatmap.
        
        Args:
            data (pd.DataFrame): Stock data
            columns (List[str], optional): Columns to include in correlation
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        if columns is None:
            # Select numeric columns
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = data[columns].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   ax=ax)
        
        ax.set_title('Correlation Matrix')
        plt.tight_layout()
        
        return fig
    
    def plot_returns_distribution(self, 
                                data: pd.DataFrame, 
                                column: str = 'Price_Change') -> go.Figure:
        """
        Plot distribution of returns.
        
        Args:
            data (pd.DataFrame): Stock data
            column (str): Column containing returns
            
        Returns:
            go.Figure: Plotly figure object
        """
        if column not in data.columns:
            logger.error(f"Column {column} not found in data")
            return go.Figure()
        
        returns = data[column].dropna()
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=50,
                name='Returns Distribution',
                opacity=0.7
            )
        )
        
        # Add normal distribution curve
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        y = y * len(returns) * (returns.max() - returns.min()) / 50  # Scale to match histogram
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='red', width=2)
            )
        )
        
        fig.update_layout(
            title='Returns Distribution',
            xaxis_title='Returns',
            yaxis_title='Frequency',
            showlegend=True
        )
        
        return fig
    
    def plot_forecast_comparison(self, 
                               actual: pd.Series, 
                               predictions: Dict[str, pd.Series],
                               title: str = 'Forecast Comparison') -> go.Figure:
        """
        Plot comparison of actual vs predicted values.
        
        Args:
            actual (pd.Series): Actual values
            predictions (Dict[str, pd.Series]): Dictionary of predictions
            title (str): Plot title
            
        Returns:
            go.Figure: Plotly figure object
        """
        fig = go.Figure()
        
        # Plot actual values
        fig.add_trace(
            go.Scatter(
                x=actual.index,
                y=actual.values,
                mode='lines',
                name='Actual',
                line=dict(color='black', width=2)
            )
        )
        
        # Plot predictions
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (model_name, pred) in enumerate(predictions.items()):
            fig.add_trace(
                go.Scatter(
                    x=pred.index,
                    y=pred.values,
                    mode='lines',
                    name=f'{model_name} (Predicted)',
                    line=dict(color=colors[i % len(colors)], width=1, dash='dash')
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Price',
            showlegend=True
        )
        
        return fig
    
    def create_dashboard(self, 
                        data: pd.DataFrame, 
                        symbol: str = '') -> go.Figure:
        """
        Create a comprehensive dashboard with multiple charts.
        
        Args:
            data (pd.DataFrame): Stock data
            symbol (str): Stock symbol
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[
                f'{symbol} Stock Price',
                'Volume',
                'RSI',
                'MACD'
            ],
            row_width=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color=self.colors['primary'])
            ),
            row=1, col=1
        )
        
        # Volume
        if 'Volume' in data.columns:
            fig.add_trace(
                go.Bar(
                    x=data['date'],
                    y=data['Volume'],
                    name='Volume',
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color=self.colors['secondary'])
                ),
                row=3, col=1
            )
            
            # Add overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # MACD
        if 'MACD' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color=self.colors['primary'])
                ),
                row=4, col=1
            )
            
            if 'MACD_Signal' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data['date'],
                        y=data['MACD_Signal'],
                        mode='lines',
                        name='MACD Signal',
                        line=dict(color=self.colors['secondary'])
                    ),
                    row=4, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Stock Dashboard',
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def save_plot(self, 
                 fig, 
                 filename: str, 
                 format: str = 'png',
                 dpi: int = 300) -> str:
        """
        Save plot to file.
        
        Args:
            fig: Figure object (matplotlib or plotly)
            filename (str): Output filename
            format (str): File format
            dpi (int): DPI for static images
            
        Returns:
            str: Path to saved file
        """
        import os
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        filepath = os.path.join('results', f"{filename}.{format}")
        
        try:
            if hasattr(fig, 'write_html'):  # Plotly figure
                fig.write_html(filepath)
            else:  # Matplotlib figure
                fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            
            logger.info(f"Plot saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving plot: {str(e)}")
            return ""

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
    
    # Initialize visualizer
    visualizer = StockVisualizer()
    
    # Create price chart
    price_chart = visualizer.plot_price_chart(sample_data, 'AAPL')
    visualizer.save_plot(price_chart, 'price_chart', 'html')
    
    print("Visualization completed!") 