import random
from datetime import datetime, timedelta
from econ_data import EconData
from article_sentiment import ArticleSentiment
from fetch_fred import download_fred_data
from fetch_stock import download_stock_data
import traceback
import pandas as pd 
from model import StockPredictor
from sklearn.preprocessing import MinMaxScaler

pd.set_option('future.no_silent_downcasting', True)

DATE_FORMAT = '%Y-%m-%d'

codes = [
    'T10YIE', 'DFF', 'DGS10', 'DEXUSEU', 'UNRATE', 'GFDEGDQ188S', 
    'A191RL1Q225SBEA', 'M2SL', 'CPIAUCSL', 'UMCSENT', 'BSCICP03USM665S', 
    'GS10', 'INDPRO', 'PAYEMS', 'PCE', 'RSAFS', 'CPATAX', 'HOUST'
]

def combine_with_sentiment(stock_data, sentiment_data):
    combined = {}
    for stock in stock_data:
        if stock in sentiment_data:
            combined[stock] = {**stock_data[stock], 'sentiment': sentiment_data[stock]}
        else:
            combined[stock] = stock_data[stock]
            print(f"No sentiment data for {stock}. Using stock data only.")
    return combined

def get_model_input(target_date, stock, num_days=15, buffer=365):
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
    elif isinstance(target_date, datetime):
        target_date = target_date.date()
    elif not isinstance(target_date, datetime.date):
        raise TypeError("target_date must be a string or a datetime.date object")
    
    start_day = target_date - timedelta(days=num_days - 1)
    buffer_day = start_day - timedelta(days=buffer)
    download_fred_data(codes, [buffer_day.strftime(DATE_FORMAT), target_date.strftime(DATE_FORMAT)])
    download_stock_data([stock], [buffer_day.strftime(DATE_FORMAT), target_date.strftime(DATE_FORMAT)])
    days = [start_day + timedelta(days=i) for i in range(num_days)]
    
    stocks = [stock]
    
    state = {stock: [] for stock in stocks}
    
    for current_day in days:
        date_str = current_day.strftime("%Y-%m-%d")
        try:
            stock_data = EconData.get_multiple_company_info(stocks, date_str)
            
            sentiment_data = ArticleSentiment.get_news_sentiment_multiple(stocks, date_str, 5)
            
            combined_data = combine_with_sentiment(stock_data, sentiment_data)
        except Exception as e:
            print(f"Error fetching data for date {date_str}: {e}")
            traceback.print_exc()
            continue
        
        for stock in stocks:
            data = combined_data.get(stock)
            if isinstance(data, dict):
                state[stock].append({'date': date_str, **data})
            else:
                print(f"Skipping stock '{stock}' on date {date_str} due to invalid data.")
    
    return state, days

def predict(model_name, inputs, key):
    model = StockPredictor(
        model_type='lstm',
        learning_rate=0.001,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        scaler=MinMaxScaler,
        attention=True,
        dropout=0.5,
        bidirectional=False,
        use_tqdm=True,
        model_name=model_name,
        input_days=15,
        predict_days=3,
    )
    output = model.predict(inputs, key)
    return output

def analyse_model_name(name:str):
    return int(name.split('_')[3].replace('Input', ''))

def get_future_data(model_name, stock_key):
    input, days = get_model_input(datetime.strftime(datetime.now(), DATE_FORMAT), stock_key, analyse_model_name(model_name))
    output = predict(model_name, input, stock_key).tolist()[0]
    stock_values = [data['Stock Value'] for data in input[stock_key]]

    return stock_values, days, output

