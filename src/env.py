from article_sentiment import ArticleSentiment
from econ_data import EconData
import random
from datetime import datetime, timedelta

class StockEnv:
    def __init__(self, stock_list, day_range):
        self.stock_list = stock_list
        self.day_range = day_range

    @staticmethod
    def combine_with_sentiment(stock_data, sentiment_data):
        for stock, sentiment_value in sentiment_data.items():
            if stock in stock_data and isinstance(stock_data[stock], dict):
                stock_data[stock]['Sentiment'] = sentiment_value
            else:
                print(f"Skipping stock '{stock}' due to unexpected data.")
        return stock_data

    @staticmethod
    def parse_date(date):
        return datetime.strptime(date, "%Y-%m-%d") if isinstance(date, str) else date

    def get_random_date(self, start_date, end_date):
        start_date = self.parse_date(start_date)
        end_date = self.parse_date(end_date)
        random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        return random_date

    def calc_state(self, input_days, output_days, specific_stock=None):
        random_day = self.get_random_date(*self.day_range)
        start_day = random_day - timedelta(days=input_days - 1)
        total_days = input_days + output_days
        days = [start_day + timedelta(days=i) for i in range(total_days)]
        if specific_stock:
            stock_list = [specific_stock]
        else:
            stock_list = self.stock_list
        state = {stock: [] for stock in stock_list}
        label = {stock: [] for stock in stock_list}
        for train_d in days[:input_days]:
            date_str = train_d.strftime("%Y-%m-%d")
            stock_data = EconData.get_multiple_company_info(stock_list, date_str)
            sentiment_data = ArticleSentiment.get_news_sentiment_multiple(stock_list, date_str, 5)
            combined_data = self.combine_with_sentiment(stock_data, sentiment_data)
            for stock in stock_list:
                data = combined_data.get(stock)
                if isinstance(data, dict):
                    state[stock].append({'date': date_str, **data})
                else:
                    print(f"Skipping stock '{stock}' on date {date_str} due to invalid data.")
        for test_d in days[input_days:]:
            date_str = test_d.strftime("%Y-%m-%d")
            stock_data = EconData.get_multiple_company_info(stock_list, date_str)
            for stock in stock_list:
                data = stock_data.get(stock)
                if isinstance(data, dict) and 'Stock Value' in data:
                    label[stock].append(data['Stock Value'])
                else:
                    print(f"Skipping stock '{stock}' on date {date_str} due to invalid data.")
        return state, label, days

    def get_stock_data(self, stock_symbol, date_str):
        stock_data = EconData.get_company_info(stock_symbol, date_str)
        return stock_data