from model import StockPredictor
from sklearn.preprocessing import MinMaxScaler
from fetch_stock import download_stock_data

stock_keys = [
    "AAPL", "MSFT", "GOOG", "V", "JNJ", "WMT", 
    "NVDA", "PG", "DIS", "MA", "HD", "VZ", "PFE", "PEP", "XOM", 
    "BAC", "MRK", "JPM", "GE", "C", "CVX", "ORCL", "IBM", "GILD"
]


date_range = ['2016-01-01', '2024-09-01']

# download_stock_data(stock_keys, date_range)

model = StockPredictor(
        model_type='lstm',
        stock_list=stock_keys,
        learning_rate=0.001,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        scaler=MinMaxScaler,
        attention=True,
        dropout=0.5,
        bidirectional=False,
        use_tqdm=True,
        input_days=15,
        predict_days=3,
)

if __name__ == '__main__':
    model.train_model(50, 64)
