import streamlit as st
import pandas as pd
import numpy as np
import os
import altair as alt
import yfinance as yf
import torch
import joblib
import sys

try:
    from src.modeling.engine import UniversalModelEngine
    from src.modeling.architecture import UniversalTransformerModel
    from src import config
    from src.model_manager import ModelManager
except (ImportError, ValueError):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.modeling.engine import UniversalModelEngine
    from src.modeling.architecture import UniversalTransformerModel
    from src import config
    from src.model_manager import ModelManager

st.set_page_config(layout="wide")
st.title("Universal Stock Trend Forecaster")
st.markdown("**PLEASE UNDER NO CIRCUMSTANCES USE THIS FOR REAL MONEY TRADING. THIS IS FOR EDUCATIONAL AND FUN PURPOSES ONLY. ALSO NOTE THAT EVEN THOUGH SOME OF THE METRICS ARE GOOD, THE BAD ONES ARE DISASTERIOUSLY BAD. THIS MODEL HAS GOOD RELIABILITY OVERALL, BUT WHEN THINGS GO WRONG, THEY REALLY GO WRONG. Beyond that, have fun :D.**")

@st.cache_resource
def load_model_manager():
    return ModelManager()

@st.cache_resource
def load_engine_with_model(model_path, scaler_path, model_config, feature_cols, tickers):
    try:
        data = pd.read_parquet(config.DATA_PATH)
        engine = UniversalModelEngine(data, model_config=model_config)
        
        engine.model = UniversalTransformerModel(
            input_dim=len(feature_cols),
            ticker_count=len(tickers),
            d_model=model_config['d_model'],
            n_heads=model_config['n_heads'],
            n_layers=model_config['n_layers'],
            dropout=model_config['dropout'],
            ticker_embedding_dim=model_config['ticker_embedding_dim'],
            sequence_length=model_config['sequence_length'],
            prediction_horizon=model_config['prediction_horizon']
        ).to(engine.device)

        state_dict = torch.load(model_path, map_location=engine.device)
        
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        engine.model.load_state_dict(state_dict, strict=False)
        engine.scaler = joblib.load(scaler_path)
        engine.feature_cols = feature_cols
        engine.tickers = tickers
        engine.ticker_map = {ticker: i for i, ticker in enumerate(tickers)}
        return engine
    except Exception as e:
        st.error("model not loading")
        return None

model_manager = load_model_manager()
saved_models = model_manager.list_saved_models()

if not saved_models:
    st.error("No trained models found. Please run the training script first.")
    st.stop()

st.sidebar.header("Model Selection")
model_options = {}
for model in saved_models:
    metadata = model['metadata']
    val_loss = metadata.get('best_val_loss')
    if val_loss is not None:
        label = f"{model['name']} (Val Loss: {val_loss:.4f})"
    else:
        label = model['name']
    model_options[label] = model

selected_model_label = st.sidebar.selectbox("Select Model", list(model_options.keys()))
selected_model = model_options[selected_model_label]

st.sidebar.subheader("Model Info")
metadata = selected_model['metadata']
st.sidebar.write(f"**Created:** {metadata.get('created_at', 'Unknown')}")
st.sidebar.write(f"**Training Time:** {metadata.get('training_time', 'Unknown')}")
st.sidebar.write(f"**Epochs:** {metadata.get('epochs_trained', 'Unknown')}")
st.sidebar.write(f"**Device:** {metadata.get('device', 'Unknown')}")

accuracy_metrics = metadata.get('accuracy_metrics', {})
if accuracy_metrics:
    st.sidebar.write("**Accuracy Metrics:**")
    for metric, value in accuracy_metrics.items():
        st.sidebar.write(f"&nbsp;&nbsp;&nbsp;**{metric.replace('_', ' ').title()}:** {value:.4f}")

model_config_loaded = metadata.get('model_config', config.MODEL_CONFIG)
feature_cols_loaded = metadata.get('feature_columns', [])
tickers_loaded = metadata.get('tickers', [])

model_path, scaler_path = model_manager.load_model_for_prediction(selected_model['path'])
engine = load_engine_with_model(model_path, scaler_path, model_config_loaded, feature_cols_loaded, tickers_loaded)

if engine:
    ticker = st.selectbox("Select a Ticker", tickers_loaded)
    
    if ticker:
        max_horizon = model_config_loaded.get('prediction_horizon', 5)
        horizon = st.slider("Select Prediction Horizon (days)", 1, 120, max_horizon)
        if st.button("Generate Forecast"):
            try:
                with st.spinner("Generating forecast..."):
                    predictions_df = engine.predict(ticker, horizon)
                    
                    hist_data = yf.Ticker(ticker).history(period="60d")
                    if hist_data.empty:
                         st.error(f"Could not fetch recent historical data for {ticker} from yfinance.")
                         st.stop()
                    last_close_price = hist_data['Close'].iloc[-1]

                    predicted_log_returns = predictions_df[config.TARGET_FEATURE]
                    predicted_prices = []
                    current_price = last_close_price
                    for log_return in predicted_log_returns:
                        current_price = current_price * np.exp(log_return)
                        predicted_prices.append(current_price)
                    
                    future_dates = pd.bdate_range(start=pd.to_datetime(hist_data.index[-1]).date(), periods=horizon+1)[1:]
                    result_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Close': predicted_prices
                    })

                    st.subheader("Forecasted Prices")
                    st.dataframe(result_df.style.format({'Predicted Close': '${:,.2f}'}))

                    st.subheader("Forecast vs. Historical Data")
                    hist_plot_df = hist_data['Close'].reset_index()
                    hist_plot_df.rename(columns={'Close': 'Price', 'Date': 'Date'}, inplace=True)
                    hist_plot_df['Type'] = 'Historical'
                    
                    pred_chart_df = result_df.rename(columns={'Predicted Close': 'Price'})
                    pred_chart_df['Type'] = 'Forecast'
                    
                    last_hist_point = hist_plot_df.iloc[[-1]].copy()
                    last_hist_point['Type'] = 'Forecast'

                    chart_df = pd.concat([hist_plot_df, last_hist_point, pred_chart_df])

                    chart = alt.Chart(chart_df).mark_line(point=True).encode(
                        x=alt.X('Date:T', title='Date'),
                        y=alt.Y('Price:Q', title='Price (USD)', scale=alt.Scale(zero=False)),
                        color='Type:N',
                        tooltip=['Date', alt.Tooltip('Price:Q', format='$.2f'), 'Type']
                    ).properties(
                        title=f"{ticker} Price Forecast"
                    ).interactive()
                    
                    st.altair_chart(chart, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during prediction or plotting: {e}")
                st.exception(e)