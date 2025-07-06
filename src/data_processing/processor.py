import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from pandas_datareader import data as pdr
import os
import datetime
import sys

try:
    from .. import config
except (ImportError, ValueError):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src import config

def _get_cache_path(name):
    return os.path.join(config.RAW_DATA_CACHE_PATH, f"{name}.parquet")

def _fetch_with_cache(name, fetch_func):
    cache_path = _get_cache_path(name)
    os.makedirs(config.RAW_DATA_CACHE_PATH, exist_ok=True)
    
    if os.path.exists(cache_path):
        cached_df = pd.read_parquet(cache_path)
        last_date = cached_df.index.max().date()
        start_date = last_date + datetime.timedelta(days=1)
    else:
        cached_df = pd.DataFrame()
        start_date = datetime.datetime.strptime(config.START_DATE, "%Y-%m-%d").date()

    end_date = datetime.date.today()
    if start_date >= end_date:
        return cached_df

    new_data = fetch_func(start_date, end_date)
    
    if not new_data.empty:
        full_df = pd.concat([cached_df, new_data])
        full_df = full_df[~full_df.index.duplicated(keep='last')]
        full_df.sort_index(inplace=True)
        full_df.to_parquet(cache_path)
        return full_df
    
    return cached_df

def _fetch_yfinance_data(start, end):
    return yf.download(config.TICKERS, start=start, end=end)

def _fetch_fred_data(start, end):
    try:
        yf.pdr_override()
    except AttributeError:
        pass
    
    df = pdr.get_data_fred(list(config.FRED_SERIES.values()), start=start, end=end)
    return df.rename(columns=dict(zip(config.FRED_SERIES.values(), config.FRED_SERIES.keys())))

def engineer_features(df):
    df_out = pd.DataFrame(index=df.index)

    for col in ['Open', 'High', 'Low', 'Close']:
        df_out[f'log_return_{col}'] = np.log(df[col] / df[col].shift(1))
    df_out['log_volume'] = np.log(df['Volume'].replace(0, 1))

    df_out['RSI_14'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df_out['MACDh_12_26_9'] = macd['MACDh_12_26_9']
    bbands = ta.bbands(df['Close'], length=20, std=2)
    for col in bbands.columns:
        df_out[col] = bbands[col]
    df_out['ATRr_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    df_out['day_of_week'] = df.index.dayofweek
    df_out['day_of_month'] = df.index.day
    df_out['week_of_year'] = df.index.isocalendar().week.astype(int)
    df_out['month'] = df.index.month
    
    return df_out

def process_data():
    if not os.path.exists('data'):
        os.makedirs('data')

    stock_df_raw = _fetch_with_cache("yfinance_raw", _fetch_yfinance_data)
    fred_df_raw = _fetch_with_cache("fred_raw", _fetch_fred_data)

    stock_df = stock_df_raw.stack(level=1, future_stack=True).rename_axis(['Date', 'ticker']).reset_index()
    fred_df = fred_df_raw.ffill().reset_index()
    fred_df = fred_df.rename(columns={'DATE': 'Date'})

    combined_df = pd.merge(stock_df, fred_df, on='Date', how='left')
    combined_df[config.MACRO_FEATURES] = combined_df.groupby('ticker')[config.MACRO_FEATURES].ffill()
    combined_df = combined_df.set_index('Date')
    
    all_features = []
    for ticker in config.TICKERS:
        ticker_df = combined_df[combined_df['ticker'] == ticker].copy()
        
        features_df = engineer_features(ticker_df)
        
        features_df['ticker'] = ticker
        for macro_col in config.MACRO_FEATURES:
            features_df[macro_col] = ticker_df[macro_col]
        
        all_features.append(features_df)
        
    final_df = pd.concat(all_features)
    final_df.dropna(inplace=True)
    final_df.to_parquet(config.DATA_PATH)

if __name__ == "__main__":
    process_data()