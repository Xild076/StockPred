import pandas as pd
from datetime import datetime, timedelta
from torch.utils.data import Dataset
import os
import numpy as np
from fetch_data import FetchFred, FetchSentiment, FetchStock
import warnings
import torch
import time

warnings.filterwarnings("ignore")

ticker_symbols = [
    "AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NFLX", "META", "NVDA", "V",
    "JPM", "JNJ", "XOM", "DIS", "BABA", "KO", "PG", "PFE", "PEP", "CSCO",
    "ORCL", "BA", "INTC", "WMT", "VZ"
]

def is_number(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

class StockDataset(Dataset):
    FEATURES = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'MACD', 'RSI',
        'FEDFUNDS', 'GS10', 'DTB3', 'DPRIME', 'DTB6', 'DTB1YR',
        'CPIAUCSL', 'PCEPI', 'CPILFESL', 'CPILFENS', 'PCEPILFE',
        'DEXUSEU', 'DEXJPUS', 'DEXUSUK', 'DTWEXBGS', 'DEXBZUS',
        'DEXCHUS', 'DEXKOUS', 'DCOILWTICO', 'GVZCLS', 'DCOILBRENTEU',
        'GDP', 'INDPRO', 'TCU', 'DGORDER', 'UNRATE', 'PAYEMS',
        'CIVPART', 'AWHMAN', 'LNS11300060', 'HOUST', 'Sentiment'
    ]
    DATE_RANGE = ['2020-01-01', '2024-11-05']

    def __init__(
        self,
        stock_list,
        day_range,
        input_days,
        output_days,
        technical_indicators=None
    ):
        self.stock_list = stock_list
        self.day_range = day_range
        self.input_days = input_days
        self.output_days = output_days
        self.total_days = input_days + output_days
        self.scaler = None
        self.label_scaler = None
        self.technical_indicators = technical_indicators if technical_indicators else []
        self.fred_data = self.load_fred_data()
        self.sentiment_data = self.load_sentiment_data()
        self.stock_data = {}
        for stock in stock_list:
            stock_df = self.load_stock_data(stock)
            self.stock_data[stock] = stock_df
        self.dates = self.create_date_list()

    def load_stock_data(self, stock):
        file_path = os.path.join('data/Stocks', f"{stock.upper()}.csv")
        if not os.path.exists(file_path):
            FetchStock.download_individual_stock_data(stock, StockDataset.DATE_RANGE)
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        data = data.reindex(pd.date_range(start=self.day_range[0], end=self.day_range[1], freq='D'))
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        data.fillna(0)
        if self.technical_indicators:
            data = self.add_technical_indicators(data)
        return data

    def add_technical_indicators(self, data):
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        data.fillna(0)
        return data

    def load_fred_data(self):
        file_path = os.path.join('data/FRED', 'fred_data.csv')
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        data.fillna(0)
        return data

    def load_sentiment_data(self):
        file_path = os.path.join('data/Sentiment', 'sentiment.csv')
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        data.fillna(0)
        return data

    def set_scaler(self, scaler, label_scaler):
        self.scaler = scaler
        self.label_scaler = label_scaler

    @staticmethod
    def parse_date(date):
        return datetime.strptime(date, "%Y-%m-%d") if isinstance(date, str) else date

    def create_date_list(self):
        start_date = self.parse_date(self.day_range[0])
        end_date = self.parse_date(self.day_range[1]) - timedelta(days=self.total_days - 1)
        date_list = []
        current_date = start_date
        while current_date <= end_date:
            date_list.append(current_date)
            current_date += timedelta(days=1)
        return date_list

    def __len__(self):
        return len(self.dates) * len(self.stock_list)

    def __getitem__(self, idx):
        stock_idx = idx % len(self.stock_list)
        date_idx = idx // len(self.stock_list)
        stock_symbol = self.stock_list[stock_idx]
        start_day = self.dates[date_idx]
        days = [start_day + timedelta(days=i) for i in range(self.total_days)]
        input_days_list = days[:self.input_days]
        output_days_list = days[self.input_days:]
        stock_input_data = self.get_stock_data(stock_symbol, input_days_list)
        fred_input_data = self.get_fred_data(input_days_list)
        sentiment_input_data = self.get_sentiment_data(input_days_list)
        input_data = self.combine_data(stock_input_data, fred_input_data, sentiment_data=sentiment_input_data)
        input_features = self.prepare_input(input_data)
        if self.scaler:
            input_features = self.scaler.transform(input_features)
        state = torch.tensor(input_features, dtype=torch.float32)
        stock_output_data = self.get_stock_data(stock_symbol, output_days_list)
        labels = [day_data['Close'] for day_data in stock_output_data]
        if self.label_scaler:
            labels_array = np.array(labels).reshape(-1, 1)
            labels_scaled = self.label_scaler.transform(labels_array)
            labels_scaled = labels_scaled.flatten()
            label = torch.tensor(labels_scaled, dtype=torch.float32)
        else:
            label = torch.tensor(labels, dtype=torch.float32)
        return state, label

    def get_stock_data(self, stock, days):
        stock_df = self.stock_data[stock]
        days_index = pd.DatetimeIndex(days)
        data = stock_df.reindex(days_index)
        data = data.ffill().bfill().fillna(0)
        data = data.to_dict('records')
        return data

    def get_fred_data(self, days):
        fred_df = self.fred_data
        days_index = pd.DatetimeIndex(days)
        data = fred_df.reindex(days_index)
        data = data.ffill().bfill().fillna(0)
        data = data.to_dict('records')
        return data

    def get_sentiment_data(self, days):
        sentiment_df = self.sentiment_data
        days_index = pd.DatetimeIndex(days)
        data = sentiment_df.reindex(days_index)
        data = data.ffill().bfill().fillna(0)
        data = data.to_dict('records')
        return data

    def combine_data(self, stock_data, fred_data, sentiment_data):
        combined_data = []
        for i in range(len(stock_data)):
            day_data = {}
            day_data.update(stock_data[i])
            day_data.update(fred_data[i])
            day_data.update(sentiment_data[i])
            combined_data.append(day_data)
        return combined_data

    def prepare_input(self, data_list):
        feature_values = []
        for day_data in data_list:
            day_features = []
            for feature in self.FEATURES:
                value = day_data.get(feature, day_data.get(feature.lower(), 0.0))
                if is_number(value):
                    day_features.append(float(value))
                else:
                    day_features.append(0.0)
            feature_values.append(day_features)
        return feature_values
    
    def prepare_input_blank(data_list):
        feature_values = []
        for day_data in data_list:
            day_features = []
            for feature in StockDataset.FEATURES:
                value = day_data.get(feature, day_data.get(feature.lower(), 0.0))
                if is_number(value):
                    day_features.append(float(value))
                else:
                    day_features.append(0.0)
            feature_values.append(day_features)
        return feature_values

    def combine_data_blank(stock_data, fred_data, sentiment_data):
        combined_data = []
        
        for day in list(stock_data):
            day_data = {}
            
            stock_day_data = stock_data[day] if isinstance(stock_data[day], dict) else {'stock': stock_data[day]}
            fred_day_data = fred_data[day] if isinstance(fred_data[day], dict) else {'fred': fred_data[day]}
            sentiment_day_data = sentiment_data[day] if isinstance(sentiment_data[day], dict) else {'sentiment': sentiment_data[day]}

            day_data.update(stock_day_data)
            day_data.update(fred_day_data)
            day_data.update(sentiment_day_data)

            combined_data.append(day_data)
        
        return combined_data