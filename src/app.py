import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
import os
import time
import altair as alt
import yfinance as yf
from model import StockPredictor

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

def type_text(text, delay=0.02, color='#BB86FC', align='center', font_size='24px'):
    placeholder = st.empty()
    typed_text = ''
    for char in text:
        typed_text += char
        placeholder.markdown(f"<p style='text-align: {align}; color: {color}; font-size:{font_size};'>{typed_text}</p>", unsafe_allow_html=True)
        time.sleep(delay)

color_schemes = {
    'Midnight Blue': {'background': '#121212', 'primary': '#BB86FC', 'secondary': '#03DAC6', 'text': '#FFFFFF'},
    'Deep Purple': {'background': '#2c003e', 'primary': '#e0b3ff', 'secondary': '#b388ff', 'text': '#FFFFFF'},
    'Teal': {'background': '#004d40', 'primary': '#b2dfdb', 'secondary': '#80cbc4', 'text': '#FFFFFF'},
    'Dark Orange': {'background': '#1a1a1a', 'primary': '#ff8c00', 'secondary': '#ffa500', 'text': '#FFFFFF'},
}

st.sidebar.header("Settings")
selected_scheme_name = st.sidebar.selectbox("Color Scheme", list(color_schemes.keys()))
selected_scheme = color_schemes[selected_scheme_name]

page_bg_css = f"""
<style>
body {{
    background-color: {selected_scheme['background']};
    color: {selected_scheme['text']};
}}
[data-testid="stSidebar"] {{
    background-color: {selected_scheme['background']};
}}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)

st.markdown(f"<h1 style='text-align: center; color: {selected_scheme['primary']}; font-size:50px;'>Stock Price Predictor</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; font-size:20px; color: {selected_scheme['text']};'>Developed by <strong>Xild076</strong></p>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #444444;'>", unsafe_allow_html=True)

MODEL_STATS_FILE_NAME = 'models/model_stats.json'

def load_model_stats() -> dict:
    if os.path.exists(MODEL_STATS_FILE_NAME):
        with open(MODEL_STATS_FILE_NAME, 'r') as json_file:
            data = json.load(json_file)
        return data
    else:
        return {}

model_stats = load_model_stats()
model_nicknames = [stats['model_nickname'] for model_name, stats in model_stats.items()]
nickname_to_model = {stats['model_nickname']: model_name for model_name, stats in model_stats.items()}

model_nicknames_with_all = ['Predict with all Models'] + model_nicknames

st.sidebar.header("Select a Model")
selected_nickname = st.sidebar.selectbox("Choose a model", model_nicknames_with_all)

if selected_nickname == 'Predict with all Models':
    selected_model_stats_list = list(model_stats.values())
    st.sidebar.subheader("Model Details")
    st.sidebar.markdown(f"<b>Number of Models:</b> <span style='color:{selected_scheme['primary']};'>{len(model_stats)}</span>", unsafe_allow_html=True)
    for stats in selected_model_stats_list:
        st.sidebar.markdown(f"<hr style='border:0.5px solid {selected_scheme['secondary']}'>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<b>Model Nickname:</b> <span style='color:{selected_scheme['primary']};'>{stats['model_nickname']}</span>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<b>Model Type:</b> <span style='color:{selected_scheme['primary']};'>{stats['model_type'].upper()}</span>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<b>Input Days:</b> <span style='color:{selected_scheme['primary']};'>{stats['input_days']}</span>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<b>Output Days:</b> <span style='color:{selected_scheme['primary']};'>{stats['output_days']}</span>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<b>Number of Features:</b> <span style='color:{selected_scheme['primary']};'>{stats['num_features']}</span>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<b>Hidden Size:</b> <span style='color:{selected_scheme['primary']};'>{stats['hidden_size']}</span>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<b>Training Date Range:</b> <span style='color:{selected_scheme['primary']};'>{stats['training_date_range'][0]} to {stats['training_date_range'][1]}</span>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<b>Epoch:</b> <span style='color:{selected_scheme['primary']};'>{stats['epoch']}</span>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<b>Best Training Loss:</b> <span style='color:{selected_scheme['primary']};'>{stats['best_train_loss']:.6f}</span>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<b>Best Validation Loss:</b> <span style='color:{selected_scheme['primary']};'>{stats['best_val_loss']:.6f}</span>", unsafe_allow_html=True)
else:
    selected_model_name = nickname_to_model[selected_nickname]
    selected_model_stats = model_stats[selected_model_name]
    st.sidebar.subheader("Model Details")
    st.sidebar.markdown(f"<b>Model Nickname:</b> <span style='color:{selected_scheme['primary']};'>{selected_model_stats['model_nickname']}</span>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<b>Model Type:</b> <span style='color:{selected_scheme['primary']};'>{selected_model_stats['model_type'].upper()}</span>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<b>Input Days:</b> <span style='color:{selected_scheme['primary']};'>{selected_model_stats['input_days']}</span>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<b>Output Days:</b> <span style='color:{selected_scheme['primary']};'>{selected_model_stats['output_days']}</span>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<b>Number of Features:</b> <span style='color:{selected_scheme['primary']};'>{selected_model_stats['num_features']}</span>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<b>Hidden Size:</b> <span style='color:{selected_scheme['primary']};'>{selected_model_stats['hidden_size']}</span>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<b>Training Date Range:</b> <span style='color:{selected_scheme['primary']};'>{selected_model_stats['training_date_range'][0]} to {selected_model_stats['training_date_range'][1]}</span>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<b>Epoch:</b> <span style='color:{selected_scheme['primary']};'>{selected_model_stats['epoch']}</span>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<b>Best Training Loss:</b> <span style='color:{selected_scheme['primary']};'>{selected_model_stats['best_train_loss']:.6f}</span>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<b>Best Validation Loss:</b> <span style='color:{selected_scheme['primary']};'>{selected_model_stats['best_val_loss']:.6f}</span>", unsafe_allow_html=True)

type_text("Predict Stock Prices", color=selected_scheme['primary'], font_size='32px')
ticker_input = st.text_input("Enter a ticker symbol (e.g., AAPL, GOOGL, AMZN)", value="")

if ticker_input:
    ticker = ticker_input.upper()
    ticker_data = yf.Ticker(ticker)
    try:
        hist = ticker_data.history(period='1y')
        if not hist.empty and len(hist) >= 2:
            current_price = hist['Close'][-1]
            previous_close = hist['Close'][-2]
            day_change = (current_price - previous_close) / previous_close * 100
            volume = hist['Volume'][-1]
        else:
            fast_info = ticker_data.fast_info
            current_price = fast_info.get('last_price', None)
            previous_close = fast_info.get('previous_close', None)
            day_change = (current_price - previous_close) / previous_close * 100 if current_price and previous_close else None
            volume = fast_info.get('volume', None)
        stock_info = ticker_data.info
        company_name = stock_info.get('shortName', ticker)
        market_cap = stock_info.get('marketCap', None)
        pe_ratio = stock_info.get('trailingPE', None)
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        company_name = ticker
        current_price = previous_close = day_change = volume = market_cap = pe_ratio = None

    st.session_state['company_name'] = company_name
    st.session_state['ticker'] = ticker
    st.session_state['current_price'] = current_price
    st.session_state['previous_close'] = previous_close
    st.session_state['day_change'] = day_change
    st.session_state['volume'] = volume
    st.session_state['market_cap'] = market_cap
    st.session_state['pe_ratio'] = pe_ratio

if 'company_name' in st.session_state:
    company_name = st.session_state['company_name']
    ticker = st.session_state['ticker']
    current_price = st.session_state.get('current_price')
    previous_close = st.session_state.get('previous_close')
    day_change = st.session_state.get('day_change')
    volume = st.session_state.get('volume')
    market_cap = st.session_state.get('market_cap')
    pe_ratio = st.session_state.get('pe_ratio')

    st.markdown(f"<h2 style='text-align: center; color: {selected_scheme['secondary']};'>{company_name} ({ticker})</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${current_price:,.2f}" if current_price else "N/A", f"{day_change:+.2f}%" if day_change else "N/A")
    col2.metric("Previous Close", f"${previous_close:,.2f}" if previous_close else "N/A")
    col3.metric("Volume", f"{volume:,}" if volume else "N/A")
    col1.metric("Market Cap", f"${market_cap:,.0f}" if market_cap else "N/A")
    col2.metric("P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio else "N/A")

if st.button("Predict"):
    if not ticker_input:
        st.error("Please enter a valid ticker symbol.")
    else:
        with st.spinner('Predicting...'):
            try:
                if selected_nickname == 'Predict with all Models':
                    predictions_list = []
                    weights = []
                    for model_name, stats in model_stats.items():
                        predictor = StockPredictor(
                            model_type=stats['model_type'],
                            stock_list=[ticker],
                            input_days=stats['input_days'],
                            output_days=stats['output_days'],
                            model_name=model_name
                        )
                        date = datetime.datetime.now().strftime('%Y-%m-%d')
                        predictions = predictor.predict(ticker, date)
                        if predictions is not None:
                            epoch = stats['epoch']
                            val_loss = stats['best_val_loss']
                            weight = (epoch / val_loss) if val_loss != 0 else epoch
                            weights.append(weight)
                            predictions_list.append(predictions['predicted_prices'])
                            past_dates = predictions['input_dates']
                            past_prices = predictions['input_prices']
                        else:
                            st.warning(f"Prediction failed with model {stats['model_nickname']}.")
                    if predictions_list:
                        common_dates = set(predictions_list[0].keys())
                        for preds in predictions_list[1:]:
                            common_dates &= set(preds.keys())
                        common_dates = sorted(list(common_dates))
                        if not common_dates:
                            st.error("No common prediction dates across models.")
                        else:
                            averaged_predictions = {}
                            total_weight = sum(weights)
                            for date_key in common_dates:
                                weighted_sum = sum(pred[date_key] * weight for pred, weight in zip(predictions_list, weights))
                                averaged_predictions[date_key] = weighted_sum / total_weight
                            st.success("Prediction complete!")
                            type_text(f"Predicted Stock Prices for {company_name} ({ticker})", color=selected_scheme['primary'], font_size='28px')
                            predicted_prices = averaged_predictions
                            predicted_df = pd.DataFrame({
                                'Date': list(predicted_prices.keys()),
                                'Predicted Price': list(predicted_prices.values())
                            })
                            predicted_df['Date'] = pd.to_datetime(predicted_df['Date'])
                            predicted_df.set_index('Date', inplace=True)
                            st.dataframe(predicted_df.style.format({'Predicted Price': '${:,.2f}'}).background_gradient(cmap='Blues'))
                            type_text("Prediction Chart", color=selected_scheme['primary'], font_size='28px')
                            future_dates = list(predicted_prices.keys())
                            future_prices = list(predicted_prices.values())
                            past_prices_list = [past_prices[date] for date in past_dates]
                            all_dates = past_dates + future_dates
                            all_prices = past_prices_list + future_prices
                            data_type = ['Past'] * len(past_dates) + ['Predicted'] * len(future_dates)
                            plot_df = pd.DataFrame({
                                'Date': all_dates,
                                'Price': all_prices,
                                'Type': data_type
                            })
                            plot_df['Date'] = pd.to_datetime(plot_df['Date'])
                            plot_df.sort_values('Date', inplace=True)
                            connection_df = pd.DataFrame({
                                'Date': [past_dates[-1], future_dates[0]],
                                'Price': [past_prices_list[-1], future_prices[0]],
                                'Type': ['Connection', 'Connection']
                            })
                            connection_df['Date'] = pd.to_datetime(connection_df['Date'])
                            base = alt.Chart(plot_df).encode(
                                x=alt.X('Date:T', axis=alt.Axis(title='Date', labelAngle=-45)),
                                y=alt.Y('Price:Q', axis=alt.Axis(title='Stock Price')),
                                tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), alt.Tooltip('Price:Q', format="$.2f"), 'Type:N']
                            )
                            line_past = base.transform_filter(
                                alt.datum.Type == 'Past'
                            ).mark_line(color=selected_scheme['primary'], strokeWidth=3, point=alt.OverlayMarkDef(color=selected_scheme['primary']))
                            line_predicted = base.transform_filter(
                                alt.datum.Type == 'Predicted'
                            ).mark_line(color=selected_scheme['secondary'], strokeDash=[5,5], strokeWidth=3, point=alt.OverlayMarkDef(color=selected_scheme['secondary']))
                            line_connection = alt.Chart(connection_df).mark_line(color=selected_scheme['secondary'], strokeDash=[5,5], strokeWidth=3).encode(
                                x='Date:T',
                                y='Price:Q'
                            )
                            chart = (line_past + line_predicted + line_connection).properties(
                                width=800,
                                height=400,
                                background=selected_scheme['background']
                            ).configure_axis(
                                labelColor=selected_scheme['text'],
                                titleColor=selected_scheme['text']
                            ).configure_title(
                                color=selected_scheme['text']
                            ).configure_view(
                                strokeWidth=0
                            )
                            st.altair_chart(chart, use_container_width=True)
                            type_text("Thank you for using the Stock Price Predictor!", color=selected_scheme['primary'], font_size='24px')
                            st.markdown(f"<p style='text-align: center; font-size:18px; color: {selected_scheme['text']};'>We hope you find these predictions insightful.</p>", unsafe_allow_html=True)
                            st.session_state['predicted_df'] = predicted_df
                            st.session_state['chart'] = chart
                    else:
                        st.error("Prediction failed with all models. Please check the ticker symbol and try again.")
                else:
                    predictor = StockPredictor(
                        model_type=selected_model_stats['model_type'],
                        stock_list=[ticker],
                        input_days=selected_model_stats['input_days'],
                        output_days=selected_model_stats['output_days'],
                        model_name=selected_model_name
                    )
                    date = datetime.datetime.now().strftime('%Y-%m-%d')
                    predictions = predictor.predict(ticker, date)
                    if predictions is not None:
                        st.success("Prediction complete!")
                        type_text(f"Predicted Stock Prices for {company_name} ({ticker})", color=selected_scheme['primary'], font_size='28px')
                        predicted_prices = predictions['predicted_prices']
                        predicted_df = pd.DataFrame({
                            'Date': list(predicted_prices.keys()),
                            'Predicted Price': list(predicted_prices.values())
                        })
                        predicted_df['Date'] = pd.to_datetime(predicted_df['Date'])
                        predicted_df.set_index('Date', inplace=True)
                        st.dataframe(predicted_df.style.format({'Predicted Price': '${:,.2f}'}).background_gradient(cmap='Blues'))
                        type_text("Prediction Chart", color=selected_scheme['primary'], font_size='28px')
                        past_dates = predictions['input_dates']
                        past_prices = predictions['input_prices']
                        future_dates = list(predicted_prices.keys())
                        future_prices = list(predicted_prices.values())
                        past_prices_list = [past_prices[date] for date in past_dates]
                        all_dates = past_dates + future_dates
                        all_prices = past_prices_list + future_prices
                        data_type = ['Past'] * len(past_dates) + ['Predicted'] * len(future_dates)
                        plot_df = pd.DataFrame({
                            'Date': all_dates,
                            'Price': all_prices,
                            'Type': data_type
                        })
                        plot_df['Date'] = pd.to_datetime(plot_df['Date'])
                        plot_df.sort_values('Date', inplace=True)
                        connection_df = pd.DataFrame({
                            'Date': [past_dates[-1], future_dates[0]],
                            'Price': [past_prices_list[-1], future_prices[0]],
                            'Type': ['Connection', 'Connection']
                        })
                        connection_df['Date'] = pd.to_datetime(connection_df['Date'])
                        base = alt.Chart(plot_df).encode(
                            x=alt.X('Date:T', axis=alt.Axis(title='Date', labelAngle=-45)),
                            y=alt.Y('Price:Q', axis=alt.Axis(title='Stock Price')),
                            tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), alt.Tooltip('Price:Q', format="$.2f"), 'Type:N']
                        )
                        line_past = base.transform_filter(
                            alt.datum.Type == 'Past'
                        ).mark_line(color=selected_scheme['primary'], strokeWidth=3, point=alt.OverlayMarkDef(color=selected_scheme['primary']))
                        line_predicted = base.transform_filter(
                            alt.datum.Type == 'Predicted'
                        ).mark_line(color=selected_scheme['secondary'], strokeDash=[5,5], strokeWidth=3, point=alt.OverlayMarkDef(color=selected_scheme['secondary']))
                        line_connection = alt.Chart(connection_df).mark_line(color=selected_scheme['secondary'], strokeDash=[5,5], strokeWidth=3).encode(
                            x='Date:T',
                            y='Price:Q'
                        )
                        chart = (line_past + line_predicted + line_connection).properties(
                            width=800,
                            height=400,
                            background=selected_scheme['background']
                        ).configure_axis(
                            labelColor=selected_scheme['text'],
                            titleColor=selected_scheme['text']
                        ).configure_title(
                            color=selected_scheme['text']
                        ).configure_view(
                            strokeWidth=0
                        )
                        st.altair_chart(chart, use_container_width=True)
                        type_text("Thank you for using the Stock Price Predictor!", color=selected_scheme['primary'], font_size='24px')
                        st.markdown(f"<p style='text-align: center; font-size:18px; color: {selected_scheme['text']};'>We hope you find these predictions insightful.</p>", unsafe_allow_html=True)
                        st.session_state['predicted_df'] = predicted_df
                        st.session_state['chart'] = chart
                    else:
                        st.error("Prediction failed. Please check the ticker symbol and try again.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    if 'predicted_df' in st.session_state:
        predicted_df = st.session_state['predicted_df']
        chart = st.session_state['chart']
        company_name = st.session_state.get('company_name', '')
        ticker = st.session_state.get('ticker', '')
        type_text(f"Predicted Stock Prices for {company_name} ({ticker})", color=selected_scheme['primary'], font_size='28px')
        st.dataframe(predicted_df.style.format({'Predicted Price': '${:,.2f}'}).background_gradient(cmap='Blues'))
        type_text("Prediction Chart", color=selected_scheme['primary'], font_size='28px')
        st.altair_chart(chart, use_container_width=True)
        type_text("Thank you for using the Stock Price Predictor!", color=selected_scheme['primary'], font_size='24px')
        st.markdown(f"<p style='text-align: center; font-size:18px; color: {selected_scheme['text']};'>We hope you find these predictions insightful.</p>", unsafe_allow_html=True)

st.markdown("<hr style='border:1px solid #444444;'>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: {selected_scheme['text']};'>Disclaimer: These predictions are AI models that predict stock prices. THEY ARE NOT ACCURATE. That being said, feel free to use this as a fun tool or a reference for investment! Happy Investing!</p>", unsafe_allow_html=True)

readme = """

# Stock Price Predictor

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
  - [Components](#components)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Usage](#usage)
  - [Running the Streamlit App](#running-the-streamlit-app)
  - [Training Models](#training-models)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Models](#models)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Overview

The **Stock Price Predictor** is a comprehensive application designed to forecast stock prices using various machine learning models. It leverages real-time data from sources like Yahoo Finance and FRED (Federal Reserve Economic Data), along with sentiment analysis derived from news articles, to provide insightful predictions. Built with **Streamlit** for an interactive user interface and **PyTorch** for robust model training and prediction, this tool is aimed at aiding investors and enthusiasts in making informed decisions.

## Features

- **Multiple Model Support:** Choose from LSTM, GRU, TCN, Transformer, and CNN models to predict stock prices.
- **Real-Time Data Integration:** Fetches and utilizes stock prices, economic indicators, and sentiment data.
- **Interactive Dashboard:** Streamlit-based UI with customizable color schemes and dynamic metrics.
- **Model Management:** Train, save, load, and compare multiple models with detailed statistics.
- **Visualization:** Displays predicted prices alongside historical data using interactive Altair charts.
- **Sentiment Analysis:** Incorporates news sentiment to enhance prediction accuracy.
- **Logging:** Comprehensive logging for monitoring and debugging.

## Architecture

The application is structured into several key components, each responsible for distinct functionalities:

### Components

1. **Streamlit App (`app.py`):**
   - **Purpose:** Serves as the user interface, allowing users to input stock tickers, select models, and view predictions.
   - **Key Features:**
     - Customizable color schemes.
     - Interactive input fields for ticker symbols.
     - Display of stock metrics (current price, volume, etc.).
     - Visualization of predicted stock prices alongside historical data.
     - Integration with model management for prediction.

2. **Data Fetching (`fetch_data.py`):**
   - **Purpose:** Handles the retrieval and preprocessing of data from various sources.
   - **Key Classes:**
     - `FetchStock`: Fetches historical stock data, financial statements, and company information using `yfinance`.
     - `FetchFred`: Retrieves economic indicators from FRED using `pandas_datareader`.
     - `FetchSentiment`: Gathers sentiment data from news articles via web scraping and performs sentiment analysis using `TextBlob`.
   - **Key Functions:**
     - `download_individual_stock_data`: Downloads and saves stock data.
     - `fetch_stock_data`: Retrieves specific day's stock data with retry mechanisms.
     - `download_fred_data`: Downloads and saves FRED data.
     - `fetch_fred_data`: Retrieves specific day's FRED data.
     - `retrieve_links`: Scrapes news articles related to a company.
     - `compute_sentiment`: Calculates sentiment polarity of text snippets.
     - `aggregate_snippet_sentiment`: Aggregates sentiment scores over dates.

3. **Model Definitions (`model_blocks.py`):**
   - **Purpose:** Defines various neural network architectures for time series prediction using PyTorch.
   - **Key Models:**
     - `LSTMAttentionModel`: Combines LSTM with an attention mechanism.
     - `GRUModel`: Utilizes Gated Recurrent Units for sequence modeling.
     - `TCNModel`: Implements a Temporal Convolutional Network.
     - `TimeSeriesTransformer`: Leverages Transformer architecture with positional encoding.
     - `CNNModel`: Applies Convolutional Neural Networks for feature extraction.
   - **Model Registry:**
     - `MODEL_BLOCKS`: A dictionary mapping model type names to their corresponding classes for easy instantiation.

4. **Model Management (`model.py`):**
   - **Purpose:** Manages the training, saving, loading, and prediction processes for models.
   - **Key Classes:**
     - `CustomLoss`: Custom loss function using Mean Squared Error (MSE).
     - `StockPredictor`: Core class handling model lifecycle, including data preparation, training, validation, testing, and prediction.
   - **Key Functionalities:**
     - **Scaler Management:** Fits and loads scalers for data normalization.
     - **Model Building:** Instantiates models based on selected type and configurations.
     - **Training Pipeline:** Handles the training loop with support for learning rate schedulers and gradient scaling.
     - **Evaluation:** Computes metrics like MAE, MSE, and RMSE, and plots results.
     - **Prediction:** Generates future stock price predictions based on input data.

5. **Dataset Handling (`stock_env.py`):**
   - **Purpose:** Prepares and processes datasets for training and prediction.
   - **Key Classes:**
     - `StockDataset`: Custom `torch.utils.data.Dataset` that combines stock data, economic indicators, and sentiment scores.
   - **Key Functionalities:**
     - **Data Loading:** Loads and preprocesses data from CSV files.
     - **Feature Engineering:** Incorporates technical indicators like MACD and RSI.
     - **Data Normalization:** Applies scaling to features and labels.
     - **Data Retrieval:** Implements methods to fetch data for specific dates and stocks.

6. **Utility Functions (`utility.py`):**
   - **Purpose:** Provides logging capabilities for monitoring and debugging.
   - **Key Classes:**
     - `Logging`: Custom logging class that writes logs to a file and optionally prints them with color-coded messages.
   - **Key Functionalities:**
     - **Logging Methods:** `success`, `error`, `alert`, and `log` methods to categorize log messages.
     - **File Handling:** Ensures logs are appended to the specified log file.

## Installation

### Prerequisites

Ensure you have the following installed on your system:

- **Python 3.8 or higher**
- **pip** (Python package manager)
- **Git** (optional, for cloning the repository)

### Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/stock-price-predictor.git
   cd stock-price-predictor

2.	**Create a Virtual Environment (Optional but Recommended)**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate


3.	**Install Dependencies**

    ```bash
    pip install -r requirements.txt

If requirements.txt is not provided, install the following packages:

    pip install streamlit pandas numpy yfinance altair torch torchvision scikit-learn matplotlib tqdm beautifulsoup4 requests newspaper3k textblob colorama pandas_datareader


4.	**Download NLTK Data (Required for TextBlob)**

    ```bash
    python -m textblob.download_corpora

## Usage

Running the Streamlit App

1.	**Navigate to the Project Directory**

    ```bash
    cd stock-price-predictor

2.	**Run the Streamlit App**

    ```bash
    streamlit run app.py


3.	**Interact with the App**
	•	Select Color Scheme: Customize the appearance from the sidebar.
	•	Select Model: Choose a specific model or predict using all available models.
	•	Enter Ticker Symbol: Input the stock ticker (e.g., AAPL, GOOGL, AMZN).
	•	View Predictions: Click “Predict” to generate and visualize predictions.

## Training Models

The application comes with pre-trained models, but you can train new models as follows:

1.	**Ensure Data is Fetched**

Before training, ensure that the necessary data is available in the data/ directories. Use the provided fetch_data.py to download stock, FRED, and sentiment data.

2.	**Run the Training Script**

    ```bash
    python model.py

This script initializes a StockPredictor instance and performs training using the specified model parameters.

3.	**Monitor Training**
The script logs progress to log.txt and displays training metrics in the console.
4.	**Access Trained Models**
Trained models are saved in the models/ directory with detailed statistics in models/model_stats.json.

Project Structure
	•	app.py: Streamlit application script for user interaction and visualization.
	•	fetch_data.py: Data fetching utilities for stocks, economic indicators, and sentiment analysis.
	•	model_blocks.py: PyTorch model definitions for various neural network architectures.
	•	model.py: Model training and prediction logic, including data preparation and evaluation.
	•	stock_env.py: Custom torch.utils.data.Dataset class for preparing datasets.
	•	utility.py: Logging utilities for monitoring and debugging.
	•	models/: Directory to store trained models, scalers, and model statistics.
	•	data/: Directory to store fetched stock data, FRED economic indicators, and sentiment data.
	•	log.txt: Log file for recording training progress and other events.
	•	README.md: Project documentation.
	•	requirements.txt: Python dependencies.

## Data Sources

	•	Stock Data: Yahoo Finance via yfinance.
	•	Economic Indicators: FRED via pandas_datareader.
	•	Sentiment Data: News articles fetched through web scraping and analyzed using TextBlob.

## Models

The application supports various neural network architectures for time series prediction:

	1.	LSTM (Long Short-Term Memory): Captures temporal dependencies in stock data.
	2.	GRU (Gated Recurrent Unit): Similar to LSTM but with a simpler structure.
	3.	TCN (Temporal Convolutional Network): Utilizes convolutional layers for temporal data.
	4.	Transformer: Leverages attention mechanisms for capturing dependencies.
	5.	CNN (Convolutional Neural Network): Applies convolutional layers for feature extraction.

Each model is defined in model_blocks.py and can be trained and evaluated independently. The StockPredictor class in model.py facilitates the training and prediction processes.

## Model Training

To train a specific model, instantiate the StockPredictor with desired parameters and call the train method. Example:

    predictor = StockPredictor(
        model_type='lstm',
        stock_list=['AAPL', 'GOOG'],
        input_days=15,
        output_days=3,
        lr=0.001,
        hidden_size=128
    )
    predictor.train(num_epochs=50, batch_size=64, lr_scheduler='rltop')

## Model Prediction

To generate predictions for a specific stock and date:

    predictions = predictor.predict(ticker='AAPL', date='2024-10-28')
    print(predictions)

## Contributing

Contributions are welcome! Please follow these steps:

1.	**Fork the Repository**
2.	**Create a Feature Branch**

    ```bash
    git checkout -b feature/YourFeature


3.	**Commit Your Changes**

    ```bash
    git commit -m "Add your feature"


4.	**Push to the Branch**

    ```bash
    git push origin feature/YourFeature


5.	**Open a Pull Request**

Provide a clear description of your changes and the problem they solve.

Acknowledgments

	•	Inspired by various stock prediction models and financial data analysis techniques.
	•	Special thanks to the developers of Streamlit, PyTorch, yfinance, TextBlob, and other open-source libraries that made this project possible.
	•	Gratitude to the contributors and the open-source community for their invaluable resources and support.

Disclaimer: *This application provides AI-driven stock price predictions. These predictions are not guaranteed to be accurate and should not be solely relied upon for investment decisions. Use this tool responsibly and at your own risk.*


"""
with st.expander("README - Project Documentation", expanded=False):
    st.markdown(readme)