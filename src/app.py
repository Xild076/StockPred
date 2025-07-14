import streamlit as st
import pandas as pd
import numpy as np
import os
import json
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
def get_model_quality(medians):
    scores = {}
    if 'median_direction_accuracy' in medians:
        v = medians['median_direction_accuracy']
        if v >= 0.85:
            scores['Direction Accuracy'] = 4
        elif v >= 0.7:
            scores['Direction Accuracy'] = 3
        elif v >= 0.55:
            scores['Direction Accuracy'] = 2
        else:
            scores['Direction Accuracy'] = 1
    if 'median_r2' in medians:
        v = medians['median_r2']
        if v >= 0.8:
            scores['R²'] = 4
        elif v >= 0.5:
            scores['R²'] = 3
        elif v >= 0:
            scores['R²'] = 2
        else:
            scores['R²'] = 1
    if 'median_mae' in medians:
        v = medians['median_mae']
        if v <= 1.0:
            scores['MAE'] = 4
        elif v <= 3.0:
            scores['MAE'] = 3
        elif v <= 10.0:
            scores['MAE'] = 2
        else:
            scores['MAE'] = 1
    if 'median_rmse' in medians:
        v = medians['median_rmse']
        if v <= 1.0:
            scores['RMSE'] = 4
        elif v <= 3.0:
            scores['RMSE'] = 3
        elif v <= 10.0:
            scores['RMSE'] = 2
        else:
            scores['RMSE'] = 1
    if 'median_mape' in medians:
        v = medians['median_mape']
        if v <= 20:
            scores['MAPE'] = 4
        elif v <= 50:
            scores['MAPE'] = 3
        elif v <= 100:
            scores['MAPE'] = 2
        else:
            scores['MAPE'] = 1
    return scores


@st.cache_resource
def load_model_manager():
    return ModelManager()

@st.cache_data
def load_data():
    return pd.read_parquet(config.DATA_PATH)

@st.cache_resource
def load_engine_with_model(model_path, scaler_path, metadata):
    try:
        model_config = metadata.get('model_config', config.MODEL_CONFIG)
        feature_cols = metadata.get('feature_columns', config.TECHNICAL_FEATURES + config.TIME_FEATURES + config.MACRO_FEATURES)
        tickers = metadata.get('tickers', config.TICKERS)
        if not feature_cols:
            feature_cols = config.TECHNICAL_FEATURES + config.TIME_FEATURES + config.MACRO_FEATURES
        if not tickers:
            tickers = config.TICKERS
        data = load_data()
        engine = UniversalModelEngine(data, model_config=model_config)
        supported_params = {
            'd_model', 'n_heads', 'n_layers', 'dropout', 'ticker_embedding_dim',
            'sequence_length', 'prediction_horizon'
        }
        filtered_config = {k: v for k, v in model_config.items() if k in supported_params}
        engine.model = UniversalTransformerModel(
            input_dim=len(feature_cols),
            ticker_count=len(tickers),
            **filtered_config
        ).to(engine.device)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=engine.device, weights_only=False)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            engine.model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if os.path.exists(scaler_path):
            engine.scaler = joblib.load(scaler_path)
        else:
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        engine.feature_cols = feature_cols
        engine.tickers = tickers
        engine.ticker_map = {ticker: i for i, ticker in enumerate(tickers)}
        return engine
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
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
    val_loss = metadata.get('best_val_loss') or metadata.get('final_val_loss')
    if val_loss is not None:
        label = f"{model['name']} (Val Loss: {val_loss:.4f})"
    else:
        label = model['name']
    model_options[label] = model

selected_model_label = st.sidebar.selectbox("Select Model", list(model_options.keys()))
selected_model = model_options[selected_model_label]

st.sidebar.subheader("Model Info")
metadata = selected_model['metadata']
st.sidebar.write(f"Created: {metadata.get('created_at', 'Unknown')}")
st.sidebar.write(f"Training Time: {metadata.get('training_time', 'Unknown')}")
st.sidebar.write(f"Epochs: {metadata.get('epochs_trained', 'Unknown')}")
st.sidebar.write(f"Device: {metadata.get('device', 'Unknown')}")
st.sidebar.write(f"Model Type: {metadata.get('model_type', 'Unknown')}")

model_config = metadata.get('model_config', {})
if model_config:
    st.sidebar.write("Model Architecture:")
    st.sidebar.write(f"d_model: {model_config.get('d_model', 'N/A')}")
    st.sidebar.write(f"n_heads: {model_config.get('n_heads', 'N/A')}")
    st.sidebar.write(f"n_layers: {model_config.get('n_layers', 'N/A')}")
    st.sidebar.write(f"sequence_length: {model_config.get('sequence_length', 'N/A')}")
    st.sidebar.write(f"prediction_horizon: {model_config.get('prediction_horizon', 'N/A')}")
    d_model = model_config.get('d_model', 256)
    n_heads = model_config.get('n_heads', 16)
    n_layers = model_config.get('n_layers', 12)
    ffn_dim = model_config.get('ffn_dim', d_model * 4)
    ticker_embedding_dim = model_config.get('ticker_embedding_dim', 32)
    feature_count = metadata.get('feature_count', 20)
    ticker_count = metadata.get('ticker_count', 7)
    embedding_params = ticker_count * ticker_embedding_dim
    input_projection_params = (feature_count + ticker_embedding_dim) * d_model + d_model
    pos_embedding_params = 1000 * d_model
    attention_params_per_layer = (d_model * d_model * 4) + (d_model * 4)
    ffn_params_per_layer = (d_model * ffn_dim) + ffn_dim + (ffn_dim * d_model) + d_model
    layer_norm_params_per_layer = d_model * 4
    transformer_params = n_layers * (attention_params_per_layer + ffn_params_per_layer + layer_norm_params_per_layer)
    output_projection_params = (d_model * (d_model // 2)) + (d_model // 2) + ((d_model // 2) * (model_config.get('prediction_horizon', 5) * feature_count)) + (model_config.get('prediction_horizon', 5) * feature_count)
    total_params = embedding_params + input_projection_params + pos_embedding_params + transformer_params + output_projection_params
    st.sidebar.write(f"Model Size: {total_params/1000000:.1f}M parameters")

train_config = metadata.get('train_config', {})
if train_config:
    st.sidebar.write("Training Config:")
    st.sidebar.write(f"Learning Rate: {train_config.get('learning_rate', 'N/A')}")
    st.sidebar.write(f"Batch Size: {train_config.get('batch_size', 'N/A')}")
    st.sidebar.write(f"Weight Decay: {train_config.get('weight_decay', 'N/A')}")

val_loss = metadata.get('best_val_loss') or metadata.get('final_val_loss')
if val_loss is not None:
    st.sidebar.write(f"Best Val Loss: {val_loss:.6f}")


def load_all_metrics(model_path, metadata):
    # Try analysis/accuracy_metrics.json
    analysis_path = os.path.join(model_path, 'analysis', 'accuracy_metrics.json')
    analysis_metrics = {}
    if os.path.exists(analysis_path):
        try:
            with open(analysis_path, 'r') as f:
                analysis_metrics = json.load(f)
        except:
            pass
    # Try metadata['accuracy_metrics']
    meta_metrics = metadata.get('accuracy_metrics', {})
    # Try top-level keys in metadata (sometimes metrics are stored directly)
    meta_top = {}
    for k in ['mae','rmse','r2','direction_accuracy','mape','median_mae','median_rmse','median_r2','median_direction_accuracy','median_mape']:
        if k in metadata:
            meta_top[k] = metadata[k]
    # Merge all, with analysis > meta_metrics > meta_top
    merged = {**meta_top, **meta_metrics, **analysis_metrics}
    # Add alias support for common alternate names
    aliases = {
        'directional_accuracy': 'direction_accuracy',
        'median_directional_accuracy': 'median_direction_accuracy',
        'median_r_squared': 'median_r2',
        'median_r2_score': 'median_r2',
        'median_mae_score': 'median_mae',
        'median_rmse_score': 'median_rmse',
        'median_mape_score': 'median_mape',
        'median_direction_acc': 'median_direction_accuracy',
        'median_r2_score': 'median_r2',
        'median_r_squared': 'median_r2',
        'median_directionacc': 'median_direction_accuracy',
        'median_mapepercent': 'median_mape',
        'medianmae': 'median_mae',
        'medianrmse': 'median_rmse',
        'medianr2': 'median_r2',
        'mediandiracc': 'median_direction_accuracy',
        'mediandirectionaccuracy': 'median_direction_accuracy',
        'medianmape': 'median_mape',
        'r2_score': 'r2',
        'r_squared': 'r2',
        'mae_score': 'mae',
        'rmse_score': 'rmse',
        'mape_score': 'mape',
    }
    for k, v in list(merged.items()):
        if k in aliases and aliases[k] not in merged:
            merged[aliases[k]] = v
    return merged

def get_metric_explanation(metric_name):
    explanations = {
        'mae': 'Mean Absolute Error: Average absolute difference between predicted and actual values. Lower is better.',
        'rmse': 'Root Mean Square Error: Square root of average squared differences. Penalizes larger errors more. Lower is better.',
        'r2': 'R-squared: Proportion of variance explained by the model. Range: -∞ to 1. Higher is better (1 = perfect).',
        'direction_accuracy': 'Direction Accuracy: Percentage of times the model correctly predicts price direction (up/down). Higher is better.',
        'mape': 'Mean Absolute Percentage Error: Average percentage error. Lower is better, but can be misleading with small values.',
        'median_mae': 'Median MAE: Median of per-sample mean absolute errors. Shows the typical error for a single prediction and is less sensitive to outliers than mean MAE.',
        'median_rmse': 'Median RMSE: Median of per-sample root mean squared errors. Shows the typical squared error for a single prediction and is less sensitive to outliers.',
        'median_r2': 'Median R²: Median of per-sample R² scores. Shows the typical explanatory power for a single prediction.',
        'median_direction_accuracy': 'Median Direction Accuracy: Median of per-sample direction accuracy. Shows the typical directional performance for a single prediction.',
        'median_mape': 'Median MAPE: Median of per-sample mean absolute percentage errors. Shows the typical percentage error for a single prediction and is less sensitive to outliers.'
    }
    return explanations.get(metric_name.lower(), 'Custom metric - see documentation for details.')


combined_metrics = load_all_metrics(selected_model['path'], metadata)
important_metrics = ['direction_accuracy', 'r2']
sidebar_metrics = {k: v for k, v in combined_metrics.items() if k.lower() in important_metrics}
other_metrics = {k: v for k, v in combined_metrics.items() if k.lower() not in important_metrics}

if sidebar_metrics:
    st.sidebar.write("Key Metrics:")
    for metric, value in sidebar_metrics.items():
        if isinstance(value, (int, float)):
            if 'accuracy' in metric.lower():
                formatted_value = f"{value:.2%}"
            elif 'r2' in metric.lower():
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = f"{value:.4f}"
            st.sidebar.write(f"{metric.replace('_', ' ').title()}: {formatted_value}")
        else:
            st.sidebar.write(f"{metric.replace('_', ' ').title()}: {value}")

feature_info = metadata.get('feature_columns', [])
if feature_info:
    st.sidebar.write(f"Features: {len(feature_info)} total")
tickers_info = metadata.get('tickers', [])
if tickers_info:
    st.sidebar.write(f"Tickers: {len(tickers_info)} ({', '.join(tickers_info)})")

try:
    model_path, scaler_path, model_metadata = model_manager.load_model_for_prediction(selected_model['path'])
    engine = load_engine_with_model(model_path, scaler_path, model_metadata)
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()


if engine:
    tickers_loaded = engine.tickers
    model_config_loaded = metadata.get('model_config', config.MODEL_CONFIG)
    medians = {k: v for k, v in combined_metrics.items() if k.startswith('median_')}
    st.header("Model Metrics Overview")
    scores = get_model_quality(medians)
    col1, col2 = st.columns(2)
    with col1:
        st.write("Median Direction Accuracy:", f"{medians.get('median_direction_accuracy', 'N/A'):.2%}" if 'median_direction_accuracy' in medians else 'N/A', f"(Score: {scores.get('Direction Accuracy', 'N/A')}/4)")
        st.write("Median R²:", f"{medians.get('median_r2', 'N/A'):.4f}" if 'median_r2' in medians else 'N/A', f"(Score: {scores.get('R²', 'N/A')}/4)")
        st.write("Median MAE:", f"{medians.get('median_mae', 'N/A'):.4f}" if 'median_mae' in medians else 'N/A', f"(Score: {scores.get('MAE', 'N/A')}/4)")
        st.write("Median RMSE:", f"{medians.get('median_rmse', 'N/A'):.4f}" if 'median_rmse' in medians else 'N/A', f"(Score: {scores.get('RMSE', 'N/A')}/4)")
        st.write("Median MAPE:", f"{medians.get('median_mape', 'N/A'):.2f}%" if 'median_mape' in medians else 'N/A', f"(Score: {scores.get('MAPE', 'N/A')}/4)")
    with col2:
        overall = sum([v for v in scores.values() if isinstance(v, int)])
        st.write("Overall Model Score:", overall, "/ 20")
        if scores.get('Direction Accuracy', 0) >= 4 and scores.get('R²', 0) >= 3:
            st.write("This model is generally reliable and achieves high directional accuracy and good explanatory power.")
        elif scores.get('Direction Accuracy', 0) >= 3:
            st.write("This model has moderate reliability and captures some market direction and variance.")
        else:
            st.write("This model is not reliable for directional prediction or variance explanation.")
    ticker = st.selectbox("Select a Ticker", tickers_loaded)
    if ticker:
        max_horizon = model_config_loaded.get('prediction_horizon', 5)
        horizon = st.slider("Select Prediction Horizon (days)", 1, 30, max_horizon)
        if st.button("Generate Forecast"):
            try:
                with st.spinner("Generating forecast..."):
                    predictions_df = engine.predict(ticker, horizon, model_path, scaler_path)
                    hist_data = yf.Ticker(ticker).history(period="60d")
                    if hist_data.empty:
                        st.error(f"Could not fetch recent historical data for {ticker} from yfinance.")
                        st.stop()
                    last_close_price = hist_data['Close'].iloc[-1]
                    target_col = metadata.get('target_feature', config.TARGET_FEATURE)
                    if target_col in predictions_df.columns:
                        predicted_log_returns = predictions_df[target_col]
                    else:
                        st.error(f"Target feature '{target_col}' not found in predictions")
                        st.stop()
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
        if combined_metrics:
            st.divider()
            st.subheader("All Model Metrics")
            st.write("All available metrics for this model, including direction accuracy, R², MAE, RMSE, MAPE, and all median metrics. Median metrics show the typical error or score for a single prediction and are less sensitive to outliers than mean metrics.")
            metrics_df = pd.DataFrame([
                {
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': f"{value:.2%}" if 'accuracy' in metric.lower() 
                           else f"{value:.4f}" if 'r2' in metric.lower()
                           else f"{value:.4f}" if 'mae' in metric.lower()
                           else f"{value:.2f}" if 'rmse' in metric.lower()
                           else f"{value:.2f}%" if 'mape' in metric.lower()
                           else f"{value:.4f}",
                    'Description': get_metric_explanation(metric)
                }
                for metric, value in combined_metrics.items()
                if isinstance(value, (int, float))
            ])
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
else:
    st.error("Failed to load the selected model. Please check the model files and try again.")