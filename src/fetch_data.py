import os
import time
import json
import tempfile
import shutil
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
from dateutil.relativedelta import relativedelta
import random

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
    data = None

    @classmethod
    def load_existing_data(cls):
        os.makedirs(cls.SAVE_PATH, exist_ok=True)
        file_path = os.path.join(cls.SAVE_PATH, 'fred_data.csv')
        if os.path.exists(file_path):
            try:
                cls.data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            except Exception:
                cls.data = pd.DataFrame()
        else:
            cls.data = pd.DataFrame()

    @classmethod
    def save_to_csv(cls, df, file_path):
        temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', newline='')
        try:
            df.to_csv(temp_file.name)
            temp_file.close()
            shutil.move(temp_file.name, file_path)
        except Exception:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)

    @classmethod
    def download_fred_data(cls, codes, date_range):
        start_date, end_date = date_range
        try:
            new_data = web.get_data_fred(codes, start=start_date, end=end_date)
            new_data = new_data.reindex(pd.date_range(start=start_date, end=end_date, freq='D'))
            new_data.ffill(inplace=True)
            new_data.bfill(inplace=True)
            if new_data.isnull().values.any():
                new_data.fillna(0, inplace=True)
            if not set(codes).issubset(new_data.columns):
                missing = set(codes) - set(new_data.columns)
                raise ValueError(f"Missing FRED codes in data: {missing}")
            if cls.data is not None and not cls.data.empty:
                combined_data = pd.concat([cls.data, new_data])
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                combined_data.sort_index(inplace=True)
                cls.data = combined_data
            else:
                cls.data = new_data
            file_path = os.path.join(cls.SAVE_PATH, 'fred_data.csv')
            cls.save_to_csv(cls.data, file_path)
        except Exception:
            pass

    @classmethod
    def fetch_fred_data(cls, day):
        if cls.data is None:
            cls.load_existing_data()
        day_dt = pd.to_datetime(day)
        if cls.data.empty:
            cls.download_fred_data(cls.FRED_CODES, cls.DATE_RANGE)
        if day_dt not in cls.data.index:
            if day_dt < cls.data.index.min():
                new_start = day
                new_end = (cls.data.index.min() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            elif day_dt > cls.data.index.max():
                new_start = (cls.data.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                new_end = day
            else:
                new_start = day
                new_end = day
            cls.download_fred_data(cls.FRED_CODES, [new_start, new_end])
        if day_dt in cls.data.index:
            return cls.data.loc[day_dt].to_dict()
        else:
            return {code: 0 for code in cls.FRED_CODES}

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
    data_loaded = {}

    @classmethod
    def save_to_csv(cls, df, file_path):
        temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', newline='')
        try:
            df.to_csv(temp_file.name)
            temp_file.close()
            shutil.move(temp_file.name, file_path)
        except Exception:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)

    @classmethod
    def download_individual_stock_data(cls, code, date_range):
        try:
            stock = yf.Ticker(code)
            hist = stock.history(start=date_range[0], end=date_range[1])
            if hist.empty:
                return
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
            company_features = {k: v for k, v in info.items() if k in cls.company_info_features}
            company_features_df = pd.DataFrame([company_features], index=pd.to_datetime([date_range[0]]))
            def reindex_fill(df, features):
                if not df.columns.is_unique:
                    df = df.loc[:, ~df.columns.duplicated()]
                df = df.reindex(columns=features).fillna(0)
                df = df.infer_objects(copy=False)
                return df
            financials = reindex_fill(financials, cls.income_statement_features)
            balance_sheet = reindex_fill(balance_sheet, cls.balance_sheet_features)
            cashflow = reindex_fill(cashflow, cls.cash_flow_features)
            company_features_df = reindex_fill(company_features_df, cls.company_info_features)
            full_data = pd.concat([
                hist[['Open', 'High', 'Low', 'Close', 'Volume']],
                financials,
                balance_sheet,
                cashflow,
                company_features_df
            ], axis=1)
            if not full_data.index.is_unique:
                full_data = full_data[~full_data.index.duplicated(keep='first')]
            full_data = full_data.ffill().bfill()
            full_data.index = full_data.index.strftime('%Y-%m-%d')
            file_path = os.path.join(cls.SAVE_PATH, f"{code.upper()}.csv")
            if os.path.exists(file_path):
                existing_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                full_data = pd.concat([existing_data, full_data])
                full_data = full_data[~full_data.index.duplicated(keep='last')]
                full_data.sort_index(inplace=True)
            cls.save_to_csv(full_data, file_path)
        except Exception:
            pass

    @classmethod
    def download_stock_data(cls, codes, date_range):
        for code in codes:
            cls.download_individual_stock_data(code, date_range)

    @classmethod
    def fetch_stock_data(cls, code, day):
        file_path = os.path.join(cls.SAVE_PATH, code.upper() + '.csv')
        day_dt = pd.to_datetime(day)
        if code not in cls.data_loaded:
            cls.data_loaded[code] = pd.DataFrame()
        if not os.path.exists(file_path):
            collection_start_date = (day_dt - timedelta(days=500)).strftime('%Y-%m-%d')
            collection_date_range = [collection_start_date, day]
            cls.download_individual_stock_data(code, collection_date_range)
        try:
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            cls.data_loaded[code] = data
        except Exception:
            return {}
        if day_dt not in cls.data_loaded[code].index:
            if day_dt < cls.data_loaded[code].index.min():
                new_start_date = day
                new_end_date = (cls.data_loaded[code].index.min() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            elif day_dt > cls.data_loaded[code].index.max():
                new_start_date = (cls.data_loaded[code].index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                new_end_date = day
            else:
                new_start_date = day
                new_end_date = day
            cls.download_individual_stock_data(code, [new_start_date, new_end_date])
            try:
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                cls.data_loaded[code] = data
            except Exception:
                return {}
        if day_dt in cls.data_loaded[code].index:
            return cls.data_loaded[code].loc[day_dt].to_dict()
        else:
            return {}

class FetchSentiment:
    SAVE_PATH = 'data/Sentiment/sentiment.csv'
    links_file = 'data/Sentiment/links.json'
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
    data = None
    articles = []

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

    @classmethod
    def load_existing_links(cls):
        if os.path.exists(cls.links_file):
            try:
                with open(cls.links_file, 'r', encoding='utf-8') as f:
                    cls.articles = json.load(f)
            except Exception:
                cls.articles = []
        else:
            cls.articles = []

    @classmethod
    def save_to_csv(cls, df, file_path):
        temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', newline='')
        try:
            df.to_csv(temp_file.name, index=False)
            temp_file.close()
            shutil.move(temp_file.name, file_path)
        except Exception:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)

    @classmethod
    def split_date_range(cls, start_date_str, end_date_str, date_format='%Y-%m-%d'):
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

    @classmethod
    def extract_article_text(cls, url):
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception:
            return None

    @classmethod
    def retrieve_links(cls, query, start_date, end_date, amount=30, rate_limiter=None):
        cls.load_existing_links()
        monthly_ranges = cls.split_date_range(start_date, end_date)
        headers_list = [{"User-Agent": "Mozilla/5.0"}]
        headers = random.choice(headers_list)
        max_retries = 3
        try:
            ticker_info = yf.Ticker(query).info
            company_name = ticker_info.get('longName', query)
            company_name = company_name.replace('Corporation', '').replace('Inc.', '').replace(',', '').replace('Group', '').replace('Holding', '').replace('Limited', '').replace('The', '').strip()
        except Exception:
            company_name = query
        for month_start, month_end in tqdm(monthly_ranges, desc=query, unit='month'):
            try:
                formatted_start = datetime.strptime(month_start, "%Y-%m-%d").strftime("%m/%d/%Y")
                formatted_end = datetime.strptime(month_end, "%Y-%m-%d").strftime("%m/%d/%Y")
            except ValueError:
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
                except requests.RequestException:
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
                except (AttributeError, TypeError):
                    continue
            cls.articles.extend(news_results)
        unique_news = {article['link']: article for article in cls.articles}.values()
        unique_news = list(unique_news)
        existing_links = {article['link'] for article in cls.articles}
        new_unique_news = [article for article in unique_news if article['link'] not in existing_links]
        cls.articles = list({article['link']: article for article in cls.articles}.values())
        try:
            os.makedirs(os.path.dirname(cls.links_file), exist_ok=True)
            with open(cls.links_file, 'w', encoding='utf-8') as f:
                json.dump(cls.articles, f, indent=2, ensure_ascii=False)
        except IOError:
            pass

    @classmethod
    def compute_sentiment(cls, text):
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception:
            return None

    @classmethod
    def aggregate_snippet_sentiment(cls, output_csv='data/Sentiment/sentiment.csv', date_range=None):
        if not os.path.exists(cls.links_file):
            return
        try:
            with open(cls.links_file, "r", encoding="utf-8") as f:
                articles = json.load(f)
        except Exception:
            return
        for article in tqdm(articles, desc="Processing articles", unit="article"):
            snippet = article.get('snippet', '')
            sentiment = cls.compute_sentiment(snippet)
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
            except ValueError:
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
                cls.save_to_csv(combined_sentiment, output_csv)
            except Exception:
                cls.save_to_csv(sentiment_pivot, output_csv)
        else:
            cls.save_to_csv(sentiment_pivot, output_csv)

    @classmethod
    def download_stock_sentiment(cls, tickers, date_range, num_articles=10):
        rate_limiter = cls.RateLimiter(1)
        for ticker in tickers:
            cls.retrieve_links(ticker, date_range[0], date_range[1], amount=num_articles, rate_limiter=rate_limiter)
        cls.aggregate_snippet_sentiment(date_range=date_range)

    @classmethod
    def fetch_sentiment_data(cls, code, day, back_up_days=15, retries=0):
        file_path = cls.SAVE_PATH
        date_dt = pd.to_datetime(day)
        start_date = (date_dt - timedelta(days=back_up_days)).strftime('%Y-%m-%d')
        end_date = (date_dt + timedelta(days=1)).strftime('%Y-%m-%d')
        if not os.path.exists(file_path):
            if retries > 0:
                return 0
            cls.download_stock_sentiment([code], date_range=[start_date, end_date], num_articles=10)
            return cls.fetch_sentiment_data(code, day, back_up_days, retries=1)
        try:
            data = pd.read_csv(file_path, parse_dates=['date'])
            data.set_index('date', inplace=True)
            data.index = pd.to_datetime(data.index)
        except Exception:
            data = pd.DataFrame()
        date_range_obj = pd.date_range(start=start_date, end=end_date)
        missing_dates = set(date_range_obj.strftime('%Y-%m-%d')) - set(data.index.strftime('%Y-%m-%d'))
        if code not in data.columns or missing_dates:
            if retries > 0:
                return 0
            cls.download_stock_sentiment([code], date_range=[start_date, end_date], num_articles=10)
            return cls.fetch_sentiment_data(code, day, back_up_days, retries=1)
        try:
            ticker_data = data[code]
        except KeyError:
            return 0
        ticker_data = ticker_data.reindex(date_range_obj)
        ticker_data = ticker_data.ffill().bfill()
        sentiment_value = ticker_data.get(date_dt, 0)
        return sentiment_value