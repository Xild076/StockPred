import logging
import os
from fetch_fred import fetch_fred_data
from fetch_stock import fetch_stock_data
from datetime import timedelta, datetime

logger = logging.getLogger(__name__)

class EconData:
    @staticmethod
    def get_company_info(stock_name, date, macro_info):
        stock_info = fetch_stock_data(stock_name, date)
        combined_info = {**stock_info, **macro_info}
        return combined_info

    @staticmethod
    def get_multiple_company_info(stock_names, date):
        macro_info = fetch_fred_data(date)
        results = {}
        for stock in stock_names:
            try:
                results[stock] = EconData.get_company_info(stock, date, macro_info)
            except Exception as e:
                logger.error(f"Error fetching data for {stock}: {e}")
                results[stock] = f"Error fetching data for {stock}: {e}"
        return results

