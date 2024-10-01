from model import StockPredictor
from sklearn.preprocessing import MinMaxScaler
from fetch_stock import download_stock_data

stock_keys = [
    "AAPL", "MSFT", "GOOG", "V", "JNJ", "WMT", 
    "NVDA", "PG", "DIS", "MA", "HD", "VZ", "PFE", "PEP", "XOM", 
    "BAC", "MRK", "JPM", "GE", "C", "CVX", "ORCL", "IBM", "GILD"
]


date_range = ['2016-01-01', '2024-09-01']

# Example usage of the StockPredictor with optimal parameters
if __name__ == "__main__":
    predictor = StockPredictor(
        model_type='lstm',               # 'transformer' or 'lstm'
        stock_list=stock_keys,    # Select relevant stocks
        learning_rate=0.001,                    # Common starting point
        hidden_size=256,                        # Balanced capacity
        num_layers=4,                           # Sufficient depth
        num_heads=8,                            # Enhanced attention
        scaler=MinMaxScaler,                    # Feature scaling
        attention=True,                         # Enable attention mechanism
        dropout=0.3,                            # Regularization
        bidirectional=False,                     # For LSTM
        use_tqdm=True,                          # Progress bars
        input_days=15,                          # Look-back period
        predict_days=3,                         # Prediction horizon
        early_stopping_patience=10              # Early stopping
    )

    # Train the model
    predictor.train_model(
        num_epochs=50,                          # Number of epochs
        batch_size=64,                          # Batch size
        validation_split=0.1,                   # 10% validation
        accumulate_steps=2,                     # Gradient accumulation
        mse_weight=1.0,                         # Regression loss weight
        dir_weight=5.0                        # Direction loss weight
    )
