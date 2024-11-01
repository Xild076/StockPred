import os
import time
import json
from datetime import datetime, timedelta
from collections import deque

import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from textblob import TextBlob
from tqdm import tqdm
from bs4 import BeautifulSoup
import requests
from newspaper import Article
import logging
from dateutil.relativedelta import relativedelta
import random

ticker_symbols = [
        "AAPL",
        "MSFT",
        "GOOG",
        "AMZN",
        "TSLA",
        "NFLX",
        "META",
        "NVDA",
        "V",
        "JPM",
        "JNJ",
        "XOM",
        "DIS",
        "BABA",
        "KO",
        "PG",
        "PFE",
        "PEP",
        "CSCO",
        "ORCL",
        "BA",
        "INTC",
        "WMT",
        "VZ"
    ]
date_range = ['2020-01-01', '2024-09-01']

class FetchFred:
    SAVE_PATH = 'data/FRED/'
    DATE_RANGE = ['2016-01-01', '2024-09-29']
    FRED_CODES = [
        'FEDFUNDS', 'GS10', 'DTB3', 'DPRIME', 'DTB6', 'DTB1YR',
        'CPIAUCSL', 'PCEPI', 'CPILFESL', 'CPILFENS', 'PCEPILFE',
        'DEXUSEU', 'DEXJPUS', 'DEXUSUK', 'DTWEXBGS', 'DEXBZUS', 'DEXCHUS',
        'DEXKOUS', 'DCOILWTICO', 'GVZCLS',
        'DCOILBRENTEU', 'GDP', 'INDPRO', 'TCU',
        'DGORDER', 'UNRATE', 'PAYEMS', 'CIVPART',
        'AWHMAN', 'LNS11300060', 'HOUST'
    ]

    @staticmethod
    def download_fred_data(codes, date_range):
        os.makedirs(FetchFred.SAVE_PATH, exist_ok=True)
        start_date, end_date = date_range
        file_path = os.path.join(FetchFred.SAVE_PATH, 'fred_data.csv')
        
        new_data = web.get_data_fred(codes, start=start_date, end=end_date)
        new_data = new_data.reindex(pd.date_range(start=start_date, end=end_date, freq='D'))
        new_data.ffill(inplace=True)
        new_data.bfill(inplace=True)
        if new_data.isnull().values.any():
            new_data.fillna(0, inplace=True)
        if not set(codes).issubset(new_data.columns):
            missing = set(codes) - set(new_data.columns)
            raise ValueError(f"Missing FRED codes in data: {missing}")

        if os.path.exists(file_path):
            existing_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            combined_data = pd.concat([existing_data, new_data])
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            combined_data.sort_index(inplace=True)
            combined_data.to_csv(file_path)
        else:
            new_data.to_csv(file_path)

    @staticmethod
    def fetch_fred_data(day):
        if isinstance(day, pd.Timestamp):
            day = day.strftime('%Y-%m-%d')
        file_path = os.path.join(FetchFred.SAVE_PATH, 'fred_data.csv')
        if not os.path.exists(file_path):
            FetchFred.download_fred_data(FetchFred.FRED_CODES, FetchFred.DATE_RANGE)
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)

        day_dt = pd.to_datetime(day)
        if day_dt not in data.index:
            min_date = data.index.min()
            max_date = data.index.max()
            if day_dt < min_date:
                new_start_date = day
                new_end_date = (min_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            elif day_dt > max_date:
                new_start_date = (max_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                new_end_date = day
            else:
                new_start_date = day
                new_end_date = day
            FetchFred.download_fred_data(FetchFred.FRED_CODES, [new_start_date, new_end_date])
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)

        if day_dt in data.index:
            return data.loc[day_dt].to_dict()
        else:
            raise ValueError(f"Data for the day {day} is not available.")


class FetchStock:
    SAVE_PATH = 'data/Stocks/'
    DATE_RANGE = ['2016-01-01', '2024-09-29']

    income_statement_features = [
        'Tax Effect Of Unusual Items',
        'Tax Rate For Calcs',
        'Normalized EBITDA',
        'EBITDA',
        'EBIT',
        'Net Income',
        'Normalized Income',
        'Net Income From Continuing And Discontinued Operation',
        'Net Income Including Noncontrolling Interests',
        'Net Income Common Stockholders',
        'Net Income Continuous Operations',
        'Total Revenue',
        'Operating Revenue',
        'Gross Profit',
        'Operating Income',
        'Total Expenses',
        'Operating Expense',
        'Reconciled Cost Of Revenue',
        'Cost Of Revenue',
        'Diluted EPS',
        'Basic EPS',
        'Interest Expense',
        'Interest Income',
        'Tax Provision'
    ]

    balance_sheet_features = [
        'Treasury Shares Number',
        'Ordinary Shares Number',
        'Share Issued',
        'Net Debt',
        'Total Debt',
        'Tangible Book Value',
        'Invested Capital',
        'Working Capital',
        'Net Tangible Assets',
        'Common Stock Equity',
        'Total Capitalization',
        'Total Equity Gross Minority Interest',
        'Stockholders Equity',
        'Retained Earnings',
        'Common Stock',
        'Total Liabilities Net Minority Interest',
        'Total Non Current Liabilities Net Minority Interest',
        'Long Term Debt',
        'Current Liabilities',
        'Total Assets',
        'Total Non Current Assets',
        'Net PPE',
        'Accumulated Depreciation',
        'Gross PPE',
        'Current Assets',
        'Inventory',
        'Accounts Receivable',
        'Cash And Cash Equivalents',
        'Cash Equivalents'
    ]

    cash_flow_features = [
        'Free Cash Flow',
        'Repurchase Of Capital Stock',
        'Repayment Of Debt',
        'Issuance Of Debt',
        'Capital Expenditure',
        'Cash Dividends Paid',
        'Operating Cash Flow',
        'Investing Cash Flow',
        'Financing Cash Flow',
        'Change In Working Capital',
        'Stock Based Compensation',
        'Depreciation And Amortization',
        'Deferred Tax',
        'Net Income From Continuing Operations'
    ]

    company_info_features = [
        'marketCap',
        'beta',
        'trailingPE',
        'forwardPE',
        'dividendYield',
        'priceToBook',
        'priceToSalesTrailing12Months',
        'revenueGrowth',
        'earningsGrowth',
        '52WeekChange',
        'SandP52WeekChange',
        'returnOnAssets',
        'returnOnEquity',
        'grossMargins',
        'operatingMargins',
        'ebitdaMargins',
        'currentRatio',
        'debtToEquity',
        'sharesOutstanding',
        'sharesShort',
        'shortRatio',
        'priceHint',
        'currentPrice',
        'targetMeanPrice',
        'recommendationMean',
        'numberOfAnalystOpinions',
        'totalCash',
        'totalDebt',
        'quickRatio',
        'totalRevenue',
        'revenuePerShare',
        'freeCashflow',
        'operatingCashflow',
        'trailingPegRatio'
    ]

    def download_individual_stock_data(code, date_range):
        os.makedirs(FetchStock.SAVE_PATH, exist_ok=True)
        try:
            stock = yf.Ticker(code)

            hist = stock.history(start=date_range[0], end=date_range[1])
            full_date_range = pd.date_range(start=date_range[0], end=date_range[1])

            hist.index = hist.index.tz_localize(None).normalize()
            hist = hist.reindex(full_date_range)

            hist[['Open', 'High', 'Low', 'Close', 'Volume']] = hist[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(method='ffill').fillna(method='bfill')

            financials = stock.financials.transpose()
            financials.index = financials.index.tz_localize(None).normalize()

            balance_sheet = stock.balance_sheet.transpose()
            balance_sheet.index = balance_sheet.index.tz_localize(None).normalize()

            cashflow = stock.cashflow.transpose()
            cashflow.index = cashflow.index.tz_localize(None).normalize()

            info = stock.info

            company_features = {k: v for k, v in info.items() if k in FetchStock.company_info_features}
            company_features_df = pd.DataFrame([company_features], index=pd.to_datetime([date_range[0]]))

            def reindex_fill(df, features):
                if not df.columns.is_unique:
                    df = df.loc[:, ~df.columns.duplicated()]
                df = df.reindex(columns=features).fillna(0)
                df = df.infer_objects(copy=False)
                return df

            financials = reindex_fill(financials, FetchStock.income_statement_features)
            balance_sheet = reindex_fill(balance_sheet, FetchStock.balance_sheet_features)
            cashflow = reindex_fill(cashflow, FetchStock.cash_flow_features)

            company_features_df = reindex_fill(company_features_df, FetchStock.company_info_features)

            full_data = pd.concat([
                hist[['Open', 'High', 'Low', 'Close', 'Volume']],
                financials,
                balance_sheet,
                cashflow,
                company_features_df
            ], axis=1)

            if not full_data.index.is_unique:
                print(f"Warning: Duplicate dates found for {code}. Dropping duplicates.")
                full_data = full_data[~full_data.index.duplicated(keep='first')]

            full_data = full_data.ffill().bfill()

            full_data.index = full_data.index.strftime('%Y-%m-%d')

            file_path = os.path.join(FetchStock.SAVE_PATH, f"{code.upper()}.csv")
            if not os.path.exists(file_path):
                with open(file_path, 'w') as file:
                    file.close()
            full_data.to_csv(file_path)
            print(f"Data saved to {file_path}")
        except Exception as e:
            print(f"Failed to download data for {code}: {e}")

    def download_stock_data(codes, date_range):
        for code in codes:
            print(f"Downloading data for {code}...")
            FetchStock.download_individual_stock_data(code, date_range)

    def fetch_stock_data(code, day):
        file_path = os.path.join(FetchStock.SAVE_PATH, code.upper() + '.csv')
        if not os.path.exists(file_path):
            print(f"{file_path} not found. Downloading data...")
            date_dt = datetime.strptime(day, '%Y-%m-%d')
            collection_start_date = (date_dt - timedelta(days=500)).strftime('%Y-%m-%d')
            collection_date_range = [collection_start_date, day]
            FetchStock.download_individual_stock_data(code, date_range=collection_date_range)
        try:
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            return {}
            
        try:
            day_data = data.loc[day]
        except KeyError:
            print(f"Date {day} not found in data for {code}. Rerunning.")
            file_path = os.path.join(FetchStock.SAVE_PATH, code.upper() + '.csv')
            date_dt = datetime.strptime(day, '%Y-%m-%d')
            collection_start_date = (date_dt - timedelta(days=500)).strftime('%Y-%m-%d')
            collection_date_range = [collection_start_date, day]
            FetchStock.download_individual_stock_data(code, date_range=collection_date_range)
            FetchStock.fetch_stock_data(code, day)
            return {}
        
        return day_data.to_dict()


class FetchSentiment:
    SAVE_PATH = 'data/Sentiment/sentiment.csv'
    TICKER_SYMBOLS = [
        "AAPL",
        "MSFT",
        "GOOG",
        "AMZN",
        "TSLA",
        "NFLX",
        "META",
        "NVDA",
        "V",
        "JPM",
        "JNJ",
        "XOM",
        "DIS",
        "BABA",
        "KO",
        "PG",
        "PFE",
        "PEP",
        "CSCO",
        "ORCL",
        "BA",
        "INTC",
        "WMT",
        "VZ"
    ]
    DATE_RANGE = ['2020-01-01', '2024-09-01']

    class RateLimiter:
        def __init__(self, max_calls, period=1.0):
            self.max_calls = max_calls
            self.period = period
            self.calls = deque()

        def acquire(self):
            now = time.time()
            while self.calls and self.calls[0] <= now - self.period:
                self.calls.popleft()
            if len(self.calls) >= self.max_calls:
                wait_time = self.period - (now - self.calls[0])
                if wait_time > 0:
                    time.sleep(wait_time)
                self.calls.popleft()
            self.calls.append(time.time())

    @staticmethod
    def split_date_range(start_date_str, end_date_str, date_format='%Y-%m-%d'):
        start_date = datetime.strptime(start_date_str, date_format)
        end_date = datetime.strptime(end_date_str, date_format)
        months = []
        current_start = start_date
        while current_start <= end_date:
            current_end = (current_start + relativedelta(months=1)) - timedelta(days=1)
            if current_end > end_date:
                current_end = end_date
            months.append((current_start.strftime(date_format),
                           current_end.strftime(date_format)))
            current_start += relativedelta(months=1)
        return months

    @staticmethod
    def extract_article_text(url):
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            return None

    @staticmethod
    def retrieve_links(query, start_date, end_date, output_metadata_file='data/Sentiment/links.json', amount=30, rate_limiter=None):
        monthly_ranges = FetchSentiment.split_date_range(start_date, end_date)
        headers_list = [{"User-Agent": "Mozilla/5.0"}]
        headers = random.choice(headers_list)
        all_news_results = []
        max_retries = 3
        try:
            ticker_info = yf.Ticker(query).info
            company_name = ticker_info.get('longName', query)
            company_name = company_name.replace('Corporation', '').replace('Inc.', '').replace(',', '').replace('Group', '').replace('Holding', '').replace('Limited', '').replace('The', '').strip()
        except Exception as e:
            company_name = query
        for month_start, month_end in tqdm(monthly_ranges, desc=query, unit='month'):
            try:
                formatted_start = datetime.strptime(month_start, "%Y-%m-%d").strftime("%m/%d/%Y")
                formatted_end = datetime.strptime(month_end, "%Y-%m-%d").strftime("%m/%d/%Y")
            except ValueError as ve:
                continue
            base_url = (
                f"https://www.google.com/search?q={requests.utils.quote(company_name)}"
                f"&gl=us&tbm=nws&num={amount}"
                f"&tbs=cdr:1,cd_min:{formatted_start},cd_max:{formatted_end}"
            )
            attempt = 0
            response = None
            while attempt < max_retries:
                try:
                    if rate_limiter:
                        rate_limiter.acquire()
                    response = requests.get(base_url, headers=headers, timeout=10)
                    response.raise_for_status()
                    break
                except requests.RequestException as e:
                    attempt += 1
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
            if response is None:
                continue
            soup = BeautifulSoup(response.content, "html.parser")
            news_results = []
            for el in soup.select("div.SoaBEf"):
                try:
                    link = el.find("a")["href"]
                    title = el.select_one("div.MBeuO").get_text()
                    snippet = el.select_one(".GI74Re").get_text()
                    date = el.select_one(".LfVVr").get_text()
                    source = el.select_one(".NUnG9d span").get_text()
                    news_results.append({
                        "link": link,
                        "title": title,
                        "snippet": snippet,
                        "date": date,
                        "source": source,
                        "ticker": query
                    })
                except (AttributeError, TypeError) as e:
                    continue
            all_news_results.extend(news_results)
        unique_news = {article['link']: article for article in all_news_results}.values()
        unique_news = list(unique_news)
        os.makedirs(os.path.dirname(output_metadata_file), exist_ok=True)
        existing_data = []
        if os.path.exists(output_metadata_file):
            try:
                if os.path.getsize(output_metadata_file) > 0:
                    with open(output_metadata_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
            except json.JSONDecodeError as jde:
                pass
            except Exception as e:
                pass
        existing_links = {article['link'] for article in existing_data}
        new_unique_news = [article for article in unique_news if article['link'] not in existing_links]
        combined_news = existing_data + new_unique_news
        combined_news = {article['link']: article for article in combined_news}.values()
        combined_news = list(combined_news)
        try:
            with open(output_metadata_file, 'w', encoding='utf-8') as f:
                json.dump(combined_news, f, indent=2, ensure_ascii=False)
        except IOError as ioe:
            pass

    @staticmethod
    def compute_sentiment(text):
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception as e:
            return None

    @staticmethod
    def aggregate_snippet_sentiment(links_file='data/Sentiment/links.json', output_csv='data/Sentiment/sentiment.csv', date_range=None):
        if not os.path.exists(links_file):
            return
        try:
            with open(links_file, "r", encoding="utf-8") as f:
                articles = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            return
        for article in tqdm(articles, desc="Processing articles", unit="article"):
            snippet = article.get('snippet', '')
            sentiment = FetchSentiment.compute_sentiment(snippet)
            article['sentiment'] = sentiment
        df = pd.DataFrame(articles)
        if 'sentiment' not in df.columns:
            return
        df = df.dropna(subset=['sentiment'])
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        if date_range:
            try:
                start_date, end_date = [datetime.strptime(d, '%Y-%m-%d') for d in date_range]
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            except ValueError as ve:
                return
        sentiment_df = df.groupby(['date', 'ticker'])['sentiment'].mean().reset_index()
        sentiment_pivot = sentiment_df.pivot_table(index='date', columns='ticker', values='sentiment', aggfunc='mean')
        if date_range:
            full_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        else:
            full_dates = pd.date_range(start=sentiment_pivot.index.min(), end=sentiment_pivot.index.max(), freq='D')
        sentiment_pivot = sentiment_pivot.reindex(full_dates)
        sentiment_pivot = sentiment_pivot.ffill().bfill()
        sentiment_pivot.index.name = 'date'
        sentiment_pivot = sentiment_pivot.reset_index()
        sentiment_pivot['date'] = sentiment_pivot['date'].dt.strftime('%Y-%m-%d')
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        if os.path.exists(output_csv):
            try:
                existing_sentiment = pd.read_csv(output_csv)
                combined_sentiment = pd.concat([existing_sentiment, sentiment_pivot])
                combined_sentiment['date'] = pd.to_datetime(combined_sentiment['date'])
                combined_sentiment = combined_sentiment.groupby(['date']).mean().reset_index()
                combined_sentiment = combined_sentiment.sort_values('date')
                combined_sentiment.to_csv(output_csv, index=False)
            except Exception as e:
                sentiment_pivot.to_csv(output_csv, index=False)
        else:
            sentiment_pivot.to_csv(output_csv, index=False)

    @staticmethod
    def download_stock_sentiment(tickers, date_range, full_text=False, num_articles=10):
        rate_limiter = FetchSentiment.RateLimiter(1)
        for ticker in tickers:
            FetchSentiment.retrieve_links(ticker, date_range[0], date_range[1], amount=num_articles, rate_limiter=rate_limiter)
        FetchSentiment.aggregate_snippet_sentiment(date_range=date_range)

    def fetch_sentiment_data(code, day, back_up_days=15, retries=0):
        file_path = FetchSentiment.SAVE_PATH
        date_dt = datetime.strptime(day, '%Y-%m-%d')
        start_date = (date_dt - timedelta(days=back_up_days)).strftime('%Y-%m-%d')
        end_date = (date_dt + timedelta(days=1)).strftime('%Y-%m-%d')

        if not os.path.exists(file_path):
            if retries > 0:
                return 0
            FetchSentiment.download_stock_sentiment([code], date_range=[start_date, end_date])
            FetchSentiment.fetch_sentiment_data(code, day, back_up_days, retries=1)
        else:
            try:
                data = pd.read_csv(file_path, parse_dates=['date'])
            except Exception as e:
                print(f"Error reading sentiment CSV: {e}")
                data = pd.DataFrame()

            date_range = pd.date_range(start=start_date, end=end_date)
            missing_dates = set(date_range.strftime('%Y-%m-%d')) - set(data['date'].astype(str))
            if code not in data.columns or missing_dates:
                if retries > 0:
                    return 0
                FetchSentiment.download_stock_sentiment([code], date_range=[start_date, end_date])
                FetchSentiment.fetch_sentiment_data(code, day, back_up_days, retries=1)

        try:
            data = pd.read_csv(file_path, parse_dates=['date'])
            data.set_index('date', inplace=True)
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            print(f"Error reading sentiment CSV after download: {e}")
            return 0

        date_range = pd.date_range(start=start_date, end=end_date)
        ticker_data = data.get(code)
        if ticker_data is None:
            print(f"No sentiment data available for ticker {code}")
            return 0

        ticker_data = ticker_data.reindex(date_range)
        ticker_data = ticker_data.ffill().bfill()

        data_updated = pd.DataFrame({code: ticker_data}, index=date_range)
        data_updated.reset_index(inplace=True)
        data_updated.rename(columns={'index': 'date'}, inplace=True)
        data_combined = pd.concat([data.reset_index(), data_updated], ignore_index=True)
        data_combined.drop_duplicates(subset=['date'], keep='last', inplace=True)
        data_combined.set_index('date', inplace=True)
        data_combined.sort_index(inplace=True)
        data_combined.to_csv(file_path)

        try:
            sentiment_value = ticker_data.loc[date_dt]
            return sentiment_value
        except KeyError:
            print(f"Sentiment data not available for {code} on {day}")
            return 0


