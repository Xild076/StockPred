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