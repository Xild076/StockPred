import os
import pickle
import yfinance as yf
from datetime import datetime
import pandas as pd
import logging

SAVE_PATH = 'data/STOCKS/'

logger = logging.getLogger("yfinance")
logger.setLevel(logging.CRITICAL)

date_range = ['2016-01-01', '2024-09-01']

def find_closest_valid_date(dates: pd.DatetimeIndex, target_date_str: str) -> pd.Timestamp:
    target_date = pd.to_datetime(target_date_str)
    valid_dates = dates.dropna()
    if not valid_dates.empty:
        closest_date = valid_dates[valid_dates.get_loc(target_date, method='nearest')]
        return closest_date
    else:
        return None

def get_financial_statement_data(stock, field_name):
    statements = {
        'financials': stock.financials,
        'balance_sheet': stock.balance_sheet,
        'cashflow': stock.cashflow
    }

    data_series_list = []

    for statement_name, statement in statements.items():
        if field_name in statement.index:
            field_series = statement.loc[field_name]
            if not field_series.empty:
                data_series_list.append(field_series)

    if data_series_list:
        data_series = pd.concat(data_series_list)
        data_series.sort_index(inplace=True)
        return data_series
    else:
        return pd.Series(dtype=float)

def download_stock_data(stocks, date_range):
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    
    for stock_name in stocks:
        stock = yf.Ticker(stock_name)
        stock_history = stock.history(start=date_range[0], end=date_range[1])
        stock_history.index = stock_history.index.tz_localize(None).normalize()
        full_date_range = pd.date_range(start=date_range[0], end=date_range[1])
        stock_history = stock_history.reindex(full_date_range)
        stock_history.ffill(inplace=True)
        stock_history.bfill(inplace=True)

        financial_fields = [
            'Net Income', 'Total Revenue', 'Gross Profit', 'Ebitda', 'Operating Income',
            'Total Assets', 'Total Liab', 'Total Debt', 'Cash',
            'Total Current Assets', 'Total Current Liabilities', 'Total Stockholder Equity',
            'Total Cash From Operating Activities', 'Free Cash Flow'
        ]

        financial_data = {}
        for field in financial_fields:
            data_series = get_financial_statement_data(stock, field)
            data_series = data_series.reindex(full_date_range)
            data_series.ffill(inplace=True)
            data_series.bfill(inplace=True)
            data_series = data_series.infer_objects()
            financial_data[field] = data_series

        info_fields = [
            'sharesOutstanding', 'dividendYield', 'marketCap', 'beta',
            'priceToBook', 'returnOnEquity'
        ]
        info_data = {}
        for field in info_fields:
            value = stock.info.get(field, None)
            if value is None or pd.isna(value):
                value = 0
            info_data[field] = value

        stock_history['MA50'] = stock_history['Close'].rolling(window=50, min_periods=1).mean()
        stock_history['MA200'] = stock_history['Close'].rolling(window=200, min_periods=1).mean()

        for date_str in full_date_range.strftime('%Y-%m-%d'):
            file_path = os.path.join(SAVE_PATH + f'{stock_name}/', date_str + '.pkl')

            close_s = stock_history['Close'][date_str]
            volume = stock_history['Volume'][date_str]

            net_income = financial_data['Net Income'][date_str]
            revenue = financial_data['Total Revenue'][date_str]
            gross_profit = financial_data['Gross Profit'][date_str]
            ebitda = financial_data['Ebitda'][date_str]
            operating_income = financial_data['Operating Income'][date_str]
            total_assets = financial_data['Total Assets'][date_str]
            total_liabilities = financial_data['Total Liab'][date_str]
            total_debt = financial_data['Total Debt'][date_str]
            cash_and_equivalents = financial_data['Cash'][date_str]
            current_assets = financial_data['Total Current Assets'][date_str]
            current_liabilities = financial_data['Total Current Liabilities'][date_str]
            total_shareholder_equity = financial_data['Total Stockholder Equity'][date_str]
            operating_cash_flow = financial_data['Total Cash From Operating Activities'][date_str]
            free_cash_flow = financial_data['Free Cash Flow'][date_str]

            shares_outstanding = info_data['sharesOutstanding']
            eps = net_income / shares_outstanding if shares_outstanding != 0 else 0
            pe_ratio = close_s / eps if eps != 0 else 0
            dividend_yield = info_data['dividendYield']
            market_cap = info_data['marketCap']
            beta = info_data['beta']
            pb_ratio = info_data['priceToBook']
            roe = info_data['returnOnEquity']

            current_ratio = current_assets / current_liabilities if current_liabilities != 0 else 0
            debt_to_equity_ratio = total_debt / total_shareholder_equity if total_shareholder_equity != 0 else 0
            profit_margin = net_income / revenue if revenue != 0 else 0
            return_on_assets = net_income / total_assets if total_assets != 0 else 0

            ma50 = stock_history['MA50'][date_str]
            ma200 = stock_history['MA200'][date_str]

            data = {
                'Stock Value': close_s,
                'Volume': volume,
                'Net Income': net_income,
                'Shares Outstanding': shares_outstanding,
                'EPS': eps,
                'PE Ratio': pe_ratio,
                'Dividend Yield': dividend_yield,
                'Market Cap': market_cap,
                'Beta': beta,
                'PB Ratio': pb_ratio,
                'ROE': roe,
                'Revenue': revenue,
                'Gross Profit': gross_profit,
                'EBITDA': ebitda,
                'Operating Income': operating_income,
                'Total Assets': total_assets,
                'Total Liabilities': total_liabilities,
                'Total Debt': total_debt,
                'Cash and Cash Equivalents': cash_and_equivalents,
                'Current Assets': current_assets,
                'Current Liabilities': current_liabilities,
                'Operating Cash Flow': operating_cash_flow,
                'Free Cash Flow': free_cash_flow,
                'Current Ratio': current_ratio,
                'Debt to Equity Ratio': debt_to_equity_ratio,
                'Profit Margin': profit_margin,
                'Return on Assets': return_on_assets,
                'MA50': ma50,
                'MA200': ma200
            }

            data = {k: (0 if pd.isna(v) or v == float('inf') or v == float('-inf') else v) for k, v in data.items()}

            if not os.path.exists(SAVE_PATH + f'{stock_name}/'):
                os.makedirs(SAVE_PATH + f'{stock_name}/')
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)

        print(f"{stock_name} Downloaded")

def fetch_stock_data(stock, day):
    file_path = os.path.join(SAVE_PATH + f'{stock}/', day + '.pkl')

    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            fred_data = pickle.load(f)
        return fred_data
    else:
        raise FileNotFoundError(f"No data found for {stock} on the date: {day}")

stock_keys = [
    "AAPL", "MSFT", "GOOG", "V", "JNJ", "WMT", 
    "NVDA", "PG", "DIS", "MA", "HD", "VZ", "PFE", "PEP", "XOM", 
    "BAC", "MRK", "JPM", "GE", "C", "CVX", "ORCL", "IBM", "GILD"
]

