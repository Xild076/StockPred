import os
import json
import time
import socket
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from textblob import TextBlob
import xml.etree.ElementTree as ET

class ArticleSentiment:
    sentiment_cache = {}
    cache_file_path = os.path.join('data', 'Sentiment', 'sentiment_cache.json')
    max_requests_per_second = 10
    last_request_time = time.time()

    @staticmethod
    def load_cache():
        if os.path.exists(ArticleSentiment.cache_file_path):
            try:
                with open(ArticleSentiment.cache_file_path, 'r') as f:
                    ArticleSentiment.sentiment_cache = json.load(f)
            except json.JSONDecodeError:
                print("Warning: sentiment_cache.json is corrupted. Resetting the cache.")
                ArticleSentiment.sentiment_cache = {}
        else:
            ArticleSentiment.sentiment_cache = {}

    @staticmethod
    def save_cache():
        os.makedirs(os.path.dirname(ArticleSentiment.cache_file_path), exist_ok=True)
        temp_file_path = ArticleSentiment.cache_file_path + ".tmp"
        with open(temp_file_path, 'w') as f:
            json.dump(ArticleSentiment.sentiment_cache, f)
        os.replace(temp_file_path, ArticleSentiment.cache_file_path)

    @staticmethod
    def check_internet_connection():
        try:
            socket.create_connection(("www.google.com", 80), timeout=5)
            return True
        except OSError:
            return False

    @staticmethod
    def throttle_request():
        current_time = time.time()
        elapsed_time = current_time - ArticleSentiment.last_request_time
        if elapsed_time < (1 / ArticleSentiment.max_requests_per_second):
            time.sleep((1 / ArticleSentiment.max_requests_per_second) - elapsed_time)
        ArticleSentiment.last_request_time = time.time()

    @staticmethod
    def get_company_news_sentiment(company_name, date, num_articles):
        if not ArticleSentiment.sentiment_cache:
            ArticleSentiment.load_cache()
        cache_key = f"{company_name}_{date}"
        if cache_key in ArticleSentiment.sentiment_cache:
            return ArticleSentiment.sentiment_cache[cache_key]
        while not ArticleSentiment.check_internet_connection():
            time.sleep(5)
        articles = []
        sources = ['gdelt_api', 'newsapi', 'mediastack_api', 'new_york_times_api', 'google_rss']
        headers = {'User-Agent': 'Mozilla/5.0'}

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_source = {}
            for source in sources:
                if source == 'gdelt_api':
                    future = executor.submit(ArticleSentiment.fetch_from_gdelt_api, company_name, date, num_articles)
                elif source == 'newsapi':
                    future = executor.submit(ArticleSentiment.fetch_from_newsapi, company_name, date, num_articles)
                elif source == 'mediastack_api':
                    future = executor.submit(ArticleSentiment.fetch_from_mediastack_api, company_name, date, num_articles)
                elif source == 'new_york_times_api':
                    future = executor.submit(ArticleSentiment.fetch_from_new_york_times_api, company_name, date, num_articles)
                elif source == 'google_rss':
                    future = executor.submit(ArticleSentiment.fetch_from_google_rss, company_name, date, num_articles, headers)
                future_to_source[future] = source

            for future in as_completed(future_to_source):
                try:
                    articles = future.result()
                    if articles:
                        break
                except Exception as e:
                    continue

        if not articles:
            return f"No articles found for {company_name} on {date}."
        
        ArticleSentiment.throttle_request()

        sentiments = [TextBlob(description).sentiment.polarity for description in articles if description]
        total_sentiment = sum(sentiments)
        count = len(sentiments)
        average_sentiment = total_sentiment / count if count > 0 else 0
        ArticleSentiment.sentiment_cache[cache_key] = average_sentiment
        ArticleSentiment.save_cache()
        return average_sentiment

    @staticmethod
    def fetch_from_google_rss(company_name, date, num_articles, headers):
        articles = []
        formatted_date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")
        search_query = f"{company_name} after:{formatted_date}"
        base_url = 'https://news.google.com/rss/search?'
        rss_url = f"{base_url}q={search_query}&hl=en-US&gl=US&ceid=US:en"
        try:
            response = requests.get(rss_url, headers=headers, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            rss_articles = root.findall(".//item")[:num_articles]
            for article in rss_articles:
                description = article.find('description').text if article.find('description') is not None else ''
                articles.append(description)
        except:
            articles = []
        return articles

    @staticmethod
    def fetch_from_gdelt_api(company_name, date, num_articles):
        articles = []
        formatted_date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")
        gdelt_base_url = 'https://api.gdeltproject.org/api/v2/doc/doc'
        gdelt_params = {
            'query': f'"{company_name}"',
            'mode': 'artlist',
            'maxrecords': num_articles,
            'format': 'json',
            'startdatetime': f"{formatted_date.replace('-', '')}000000",
            'sort': 'date'
        }
        try:
            response = requests.get(gdelt_base_url, params=gdelt_params, timeout=10)
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '')
            if 'application/json' in content_type:
                gdelt_data = response.json()
                for article in gdelt_data.get('articles', []):
                    articles.append(article.get('seendescription', ''))
        except:
            articles = []
        return articles

    @staticmethod
    def fetch_from_newsapi(company_name, date, num_articles):
        articles = []
        api_key = 'f713d89c25234c088fb1ff97be841c9d' 
        newsapi_url = 'https://newsapi.org/v2/everything'
        params = {
            'q': company_name,
            'from': date,
            'sortBy': 'publishedAt',
            'pageSize': num_articles,
            'apiKey': api_key,
            'language': 'en',
        }
        try:
            response = requests.get(newsapi_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            for article in data.get('articles', []):
                description = article.get('description', '')
                articles.append(description)
        except:
            articles = []
        return articles

    @staticmethod
    def fetch_from_mediastack_api(company_name, date, num_articles):
        articles = []
        api_key = 'b039a23b4277eebde246908c46f628ac'
        mediastack_url = 'http://api.mediastack.com/v1/news'
        params = {
            'access_key': api_key,
            'keywords': company_name,
            'languages': 'en',
            'date': date,
            'limit': num_articles,
        }
        try:
            response = requests.get(mediastack_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            for article in data.get('data', []):
                description = article.get('description', '')
                articles.append(description)
        except:
            articles = []
        return articles

    @staticmethod
    def fetch_from_new_york_times_api(company_name, date, num_articles):
        articles = []
        api_key = 'cHlKrcnZXNBt4TwO2R4PeUjkznd5pYhf'
        nyt_url = 'https://api.nytimes.com/svc/search/v2/articlesearch.json'
        params = {
            'q': company_name,
            'begin_date': date.replace('-', ''),
            'sort': 'newest',
            'api-key': api_key,
            'language': 'en',
            'page': 0
        }
        try:
            response = requests.get(nyt_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            docs = data.get('response', {}).get('docs', [])[:num_articles]
            for doc in docs:
                description = doc.get('abstract', '')
                articles.append(description)
        except:
            articles = []
        return articles

    @staticmethod
    def get_news_sentiment_multiple(companies, date, num_articles):
        results = {}
        max_workers = 10
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                company: executor.submit(
                    ArticleSentiment.get_company_news_sentiment,
                    company, date, num_articles
                ) for company in companies
            }
            for company, future in futures.items():
                try:
                    results[company] = future.result()
                except:
                    results[company] = None
        return results
    
    @staticmethod
    def get_sentiment_for_date_range(companies, start_date, end_date, num_articles):
        ArticleSentiment.load_cache()

        results = {}
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        current_date = start
        
        while current_date <= end:
            formatted_date = current_date.strftime("%Y-%m-%d")
            print(f"Processing sentiment for {formatted_date}")
            for company in companies:
                cache_key = f"{company}_{formatted_date}"

                if cache_key in ArticleSentiment.sentiment_cache:
                    print(f"Sentiment for {company} on {formatted_date} is already cached.")
                    sentiment = ArticleSentiment.sentiment_cache[cache_key]
                else:
                    sentiment = ArticleSentiment.get_company_news_sentiment(company, formatted_date, num_articles)
                    ArticleSentiment.sentiment_cache[cache_key] = sentiment

                if company not in results:
                    results[company] = {}
                results[company][formatted_date] = sentiment

            current_date += timedelta(days=1)

        ArticleSentiment.save_cache()
        return results

"""stock_keys = [
    "AAPL", "MSFT", "GOOG", "V", "JNJ", "WMT", 
    "NVDA", "PG", "DIS", "MA", "HD", "VZ", "PFE", "PEP", "XOM", 
    "BAC", "MRK", "JPM", "GE", "C", "CVX", "ORCL", "IBM", "GILD"
]

date_range = ['2018-01-01', '2024-09-01']

ArticleSentiment.get_sentiment_for_date_range(stock_keys, date_range[0], date_range[1], 5)"""

date_range = ['2024-09-26', '2024-09-29']

