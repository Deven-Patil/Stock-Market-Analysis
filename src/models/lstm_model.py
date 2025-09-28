"""
LSTM Model for Stock Price Forecasting

This module implements Long Short-Term Memory (LSTM) neural networks
for time series forecasting of stock prices.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMModel:
    """
    A class to implement LSTM models for stock price forecasting.
    """
    
    def __init__(self, 
                 sequence_length: int = 60,
                 n_features: int = 1,
                 n_lstm_layers: int = 2,
                 n_dense_layers: int = 1,
                 lstm_units: int = 50,
                 dense_units: int = 25,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        """
        Initialize the LSTM model.
        
        Args:
            sequence_length (int): Number of time steps to look back
            n_features (int): Number of features
            n_lstm_layers (int): Number of LSTM layers
            n_dense_layers (int): Number of dense layers
            lstm_units (int): Number of units in LSTM layers
            dense_units (int): Number of units in dense layers
            dropout_rate (float): Dropout rate
            learning_rate (float): Learning rate
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_lstm_layers = n_lstm_layers
        self.n_dense_layers = n_dense_layers
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.scaler = MinMaxScaler()
        self.model = None
        self.history = None
        self.is_fitted = False
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def prepare_sequences(self, 
                         data: pd.DataFrame, 
                         target_column: str = 'Close',
                         feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Target column name
            feature_columns (List[str], optional): Feature columns to use
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X and y arrays for training
        """
        logger.info("Preparing sequences for LSTM")
        
        # Select features
        if feature_columns is None:
            feature_columns = [target_column]
        
        # Ensure target column is included
        if target_column not in feature_columns:
            feature_columns.append(target_column)
        
        # Select data
        selected_data = data[feature_columns].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(selected_data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, feature_columns.index(target_column)])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Sequences prepared - X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape (Tuple[int, int]): Input shape (sequence_length, n_features)
            
        Returns:
            Sequential: Compiled Keras model
        """
        logger.info("Building LSTM model")
        
        model = Sequential()
        
        # First LSTM layer
        if self.n_lstm_layers == 1:
            model.add(LSTM(
                units=self.lstm_units,
                return_sequences=False,
                input_shape=input_shape
            ))
        else:
            model.add(LSTM(
                units=self.lstm_units,
                return_sequences=True,
                input_shape=input_shape
            ))
            model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i in range(1, self.n_lstm_layers):
            if i == self.n_lstm_layers - 1:
                # Last LSTM layer
                model.add(LSTM(units=self.lstm_units, return_sequences=False))
            else:
                model.add(LSTM(units=self.lstm_units, return_sequences=True))
            model.add(Dropout(self.dropout_rate))
        
        # Dense layers
        for i in range(self.n_dense_layers):
            model.add(Dense(units=self.dense_units, activation='relu'))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(units=1, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("LSTM model built successfully")
        return model
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            validation_split: float = 0.2,
            verbose: int = 1) -> Dict[str, Any]:
        """
        Fit the LSTM model to the training data.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_val (np.ndarray, optional): Validation features
            y_val (np.ndarray, optional): Validation targets
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            validation_split (float): Validation split if no validation data provided
            verbose (int): Verbosity level
            
        Returns:
            Dict[str, Any]: Training history and model info
        """
        logger.info("Training LSTM model")
        
        # Build model
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=verbose
            )
        ]
        
        # Train model
        if X_val is not None and y_val is not None:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
        else:
            self.history = self.model.fit(
                X_train, y_train,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
        
        self.is_fitted = True
        
        # Get training results
        results = {
            'model': self.model,
            'history': self.history.history,
            'final_loss': self.history.history['loss'][-1],
            'final_val_loss': self.history.history['val_loss'][-1] if 'val_loss' in self.history.history else None
        }
        
        logger.info(f"LSTM model training completed. Final loss: {results['final_loss']:.4f}")
        return results
    
    def predict(self, 
               X: np.ndarray, 
               inverse_transform: bool = True) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (np.ndarray): Input features
            inverse_transform (bool): Whether to inverse transform predictions
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = self.model.predict(X)
        
        if inverse_transform:
            # Inverse transform predictions
            # Create dummy array with same shape as original features
            dummy = np.zeros((len(predictions), self.scaler.n_features_in_))
            dummy[:, 0] = predictions.flatten()  # Assuming target is first feature
            predictions = self.scaler.inverse_transform(dummy)[:, 0]
        
        return predictions
    
    def forecast_future(self, 
                       last_sequence: np.ndarray, 
                       steps: int = 30,
                       inverse_transform: bool = True) -> np.ndarray:
        """
        Forecast future values using the trained model.
        
        Args:
            last_sequence (np.ndarray): Last sequence of data
            steps (int): Number of steps to forecast
            inverse_transform (bool): Whether to inverse transform predictions
            
        Returns:
            np.ndarray: Forecasted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making forecasts")
        
        forecasts = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            # Reshape for prediction
            X_pred = current_sequence.reshape(1, self.sequence_length, self.n_features)
            
            # Make prediction
            pred = self.model.predict(X_pred, verbose=0)
            forecasts.append(pred[0, 0])
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = pred[0]
        
        forecasts = np.array(forecasts)
        
        if inverse_transform:
            # Inverse transform forecasts
            dummy = np.zeros((len(forecasts), self.scaler.n_features_in_))
            dummy[:, 0] = forecasts
            forecasts = self.scaler.inverse_transform(dummy)[:, 0]
        
        return forecasts
    
    def evaluate(self, 
                X_test: np.ndarray, 
                y_test: np.ndarray,
                inverse_transform: bool = True) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
            inverse_transform (bool): Whether to inverse transform for evaluation
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test, inverse_transform=inverse_transform)
        
        if inverse_transform:
            # Inverse transform actual values for comparison
            dummy = np.zeros((len(y_test), self.scaler.n_features_in_))
            dummy[:, 0] = y_test
            y_test_actual = self.scaler.inverse_transform(dummy)[:, 0]
        else:
            y_test_actual = y_test
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test_actual, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test_actual, y_pred)),
            'mae': mean_absolute_error(y_test_actual, y_pred),
            'mape': np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100,
            'r2': r2_score(y_test_actual, y_pred)
        }
        
        logger.info("LSTM model evaluation:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.4f}")
        
        return metrics
    
    def plot_training_history(self) -> None:
        """Plot training history."""
        if self.history is None:
            logger.warning("No training history available")
            return
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot MAE
        if 'mae' in self.history.history:
            ax2.plot(self.history.history['mae'], label='Training MAE')
            if 'val_mae' in self.history.history:
                ax2.plot(self.history.history['val_mae'], label='Validation MAE')
            ax2.set_title('Model MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str) -> str:
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
            
        Returns:
            str: Path to saved model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        try:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return ""
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.model = tf.keras.models.load_model(filepath)
            self.is_fitted = True
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def get_model_summary(self) -> str:
        """
        Get model summary.
        
        Returns:
            str: Model summary
        """
        if self.model is None:
            return "No model available"
        
        # Capture model summary
        from io import StringIO
        summary_io = StringIO()
        self.model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
        return summary_io.getvalue()

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=500, freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'Close': np.random.randn(500).cumsum() + 100,
        'Volume': np.random.uniform(1000000, 5000000, 500)
    })
    
    # Initialize LSTM model
    lstm_model = LSTMModel(
        sequence_length=30,
        n_features=2,
        n_lstm_layers=2,
        lstm_units=50,
        dense_units=25,
        dropout_rate=0.2
    )
    
    # Prepare sequences
    X, y = lstm_model.prepare_sequences(sample_data, target_column='Close', 
                                       feature_columns=['Close', 'Volume'])
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    results = lstm_model.fit(X_train, y_train, epochs=50, batch_size=32)
    
    # Evaluate model
    metrics = lstm_model.evaluate(X_test, y_test)
    
    print("LSTM model completed!") 