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
        hidden_size=128,                        # Balanced capacity
        num_layers=4,                           # Sufficient depth
        num_heads=8,                            # Enhanced attention
        scaler=MinMaxScaler,                    # Feature scaling
        attention=False,                         # Enable attention mechanism
        dropout=0.3,                            # Regularization
        bidirectional=False,                     # For LSTM
        use_tqdm=True,                          # Progress bars
        input_days=15,                          # Look-back period
        predict_days=3,                         # Prediction horizon
        early_stopping_patience=10,              # Early stopping
    )

    predictor.train_model(50, 64)
