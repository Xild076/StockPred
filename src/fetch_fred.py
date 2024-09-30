import pandas as pd
import os
import time
import pandas_datareader as pdr
from datetime import datetime, timedelta
import pickle

SAVE_PATH = 'data/FRED/'

date_range = ['2016-01-01', '2024-09-29']

codes = ['T10YIE', 'DFF', 'DGS10', 'DEXUSEU', 'UNRATE', 'GFDEGDQ188S', 
        'A191RL1Q225SBEA', 'M2SL', 'CPIAUCSL', 'UMCSENT', 'BSCICP03USM665S', 
        'GS10', 'INDPRO', 'PAYEMS', 'PCE', 'RSAFS', 'CPATAX', 'HOUST']

def download_fred_data(codes, date_range):
    datas = pdr.get_data_fred(codes, start=date_range[0], end=date_range[1])
    full_date_range = pd.date_range(start=date_range[0], end=date_range[1])
    datas = datas.reindex(full_date_range)
    datas.ffill(inplace=True)
    datas.bfill(inplace=True)
    date_dt_s = datetime.strptime(date_range[0], '%Y-%m-%d')
    date_dt_e = datetime.strptime(date_range[1], '%Y-%m-%d')

    while date_dt_s <= date_dt_e:
        date_str = datetime.strftime(date_dt_s, '%Y-%m-%d')
        file_path = os.path.join(SAVE_PATH, date_str + '.pkl')
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        with open(file_path, 'wb') as f:
            pickle.dump(datas.loc[date_str].to_dict(), f)
        date_dt_s += timedelta(days=1)

def fetch_fred_data(day):
    file_path = os.path.join(SAVE_PATH, day + '.pkl')

    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            fred_data = pickle.load(f)
        return fred_data
    else:
        raise FileNotFoundError(f"No data found for the date: {day}")


