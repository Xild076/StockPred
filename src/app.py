import streamlit as st
import json
import os
from fetch_fred import download_fred_data
from fetch_stock import download_stock_data
from econ_data import EconData
from article_sentiment import ArticleSentiment
from model import StockPredictor
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import traceback
import time  # For animations
from test_model import get_future_data

# Constants
DATE_FORMAT = '%Y-%m-%d'

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Stock Price Prediction App",
    layout="wide",
    page_icon="📊"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    /* Overall background */
    .css-1d391kg {
        background-color: #2E3440;
        color: #D8DEE9;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #3B4252;
    }
    
    /* Header styling */
    .css-1aumxhk p {
        font-size: 16px;
    }
    
    /* Button styling */
    .stButton>button {
        color: white;
        background-color: #81A1C1;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 14px;
    }
    
    /* Table styling */
    .dataframe {
        background-color: #3B4252;
        border-radius: 8px;
    }
    
    /* Title styling */
    h1 {
        color: #88C0D0;
        font-family: 'Arial';
    }
    
    /* Subheader styling */
    h2, h3, h4 {
        color: #81A1C1;
    }
    </style>
    """, unsafe_allow_html=True)

# Caching Model History
@st.cache_data
def load_model_history(history_file='models/history.json'):
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
        return history
    else:
        return {}

# Function to display model specifications using a Markdown table
def display_model_specs(model_data):
    if not model_data:
        st.info("No specifications available for this model.")
        return
    
    st.subheader("Model Specifications")
    
    # Prepare specifications
    specs = {
        "Nickname": model_data.get('nickname', 'N/A'),
        "Model Type": model_data.get('model_type', 'N/A'),
        "Epochs": model_data.get('epoch', 'N/A'),
        "Train Loss": model_data.get('train_loss', 'N/A'),
        "Validation Loss": model_data.get('val_loss', 'N/A'),
        "Learning Rate": model_data.get('lr', 'N/A'),
        "Hidden Size": model_data.get('hidden_size', 'N/A'),
        "Number of Layers": model_data.get('num_layers', 'N/A'),
        "Number of Heads": model_data.get('num_heads', 'N/A'),
        "Dropout": model_data.get('dropout', 'N/A'),
        "Bidirectional": model_data.get('bidirectional', 'N/A'),
        "Attention": model_data.get('attention', 'N/A'),
        "Input Days": model_data.get('input_days', 'N/A'),
        "Predict Days": model_data.get('predict_days', 'N/A'),
    }
    
    # Create Markdown table without index
    markdown_table = "| Specification | Value |\n|---------------|-------|\n"
    for spec, value in specs.items():
        markdown_table += f"| {spec} | {value} |\n"
    
    st.markdown(markdown_table, unsafe_allow_html=True)

# Function to split data into segments based on price movement
def get_price_segments(dates, prices):
    segments = []
    if len(prices) < 2:
        return segments
    
    current_segment = {
        "x": [dates[0]],
        "y": [prices[0]],
        "color": "green" if prices[1] >= prices[0] else "red"
    }
    
    for i in range(1, len(prices)):
        direction = "green" if prices[i] >= prices[i-1] else "red"
        if direction == current_segment["color"]:
            current_segment["x"].append(dates[i])
            current_segment["y"].append(prices[i])
        else:
            segments.append(current_segment)
            current_segment = {
                "x": [dates[i-1], dates[i]],
                "y": [prices[i-1], prices[i]],
                "color": direction
            }
    segments.append(current_segment)
    return segments

# Function to generate a unique nickname
def generate_unique_nickname(existing_nicknames):
    import random
    adjectives = [
        "Swift", "Quantum", "Eagle", "Pioneer", "Vivid",
        "Silent", "Dynamic", "Radiant", "Luminous", "Brisk",
        "Nimble", "Sage", "Vibrant", "Graceful", "Stellar"
    ]
    nouns = [
        "Falcon", "Phoenix", "Sentinel", "Voyager", "Nimbus",
        "Aurora", "Orion", "Zenith", "Nova", "Echo",
        "Comet", "Specter", "Mirage", "Pulse", "Blaze"
    ]
    while True:
        adjective = random.choice(adjectives)
        noun = random.choice(nouns)
        nickname = f"{adjective} {noun}"
        if nickname not in existing_nicknames:
            return nickname

# Function for animated title (typing effect)
def animated_title(title_text):
    title_placeholder = st.empty()
    for i in range(1, len(title_text)+1):
        title_placeholder.markdown(f"# {title_text[:i]}")
        time.sleep(0.05)

# Main Application
def main():
    # Animated Title
    animated_title("Stock Price Prediction App")
    st.markdown("""
    Welcome to the **Stock Price Prediction App**! Select a trained model by its nickname, input a stock ticker symbol, and visualize the projected stock prices along with insightful metrics.
    """)
    
    # Sidebar Configuration
    st.sidebar.header("Configuration")
    
    # Load Model History
    history = load_model_history()
    if not history:
        st.sidebar.error("No models found in history. Please train a model first.")
        st.stop()
    
    # Extract nicknames and ensure they are unique
    model_nicknames = {}
    existing_nicknames = set()
    for model_id, model_data in history.items():
        nickname = model_data.get('nickname', '')
        if nickname:
            if nickname in existing_nicknames:
                # Handle duplicate nicknames by appending model_id
                nickname = f"{nickname} ({model_id})"
            model_nicknames[nickname] = model_id
            existing_nicknames.add(nickname)
        else:
            # Generate a unique nickname if missing
            nickname = generate_unique_nickname(existing_nicknames)
            model_nicknames[nickname] = model_id
            existing_nicknames.add(nickname)
    
    # Model Search Functionality: Search by nickname only
    search_query = st.sidebar.text_input("Search Models by Nickname", value="")
    filtered_nicknames = [nickname for nickname in model_nicknames.keys() if search_query.lower() in nickname.lower()]
    if not filtered_nicknames:
        st.sidebar.warning("No models match your search.")
        st.stop()
    
    # Select Model by Nickname
    selected_nickname = st.sidebar.selectbox("Select a Model", filtered_nicknames)
    selected_model_id = model_nicknames[selected_nickname]
    selected_model_data = history.get(selected_model_id, {})
    
    # Display Model Specifications
    display_model_specs(selected_model_data)
    
    st.sidebar.markdown("---")
    
    # Input for Stock Ticker Symbol
    stock_code = st.sidebar.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT)", value="AAPL")
    
    # Predict Button
    if st.sidebar.button("Predict"):
        if not stock_code:
            st.sidebar.error("Please enter a valid stock ticker symbol.")
        else:
            with st.spinner("Loading model and performing prediction..."):
                try:
                    stock_code_upper = stock_code.upper()
                    stock_values, days, prediction = get_future_data(selected_model_id, stock_code_upper)
                    
                    # Prepare Dates
                    past_dates = [day.strftime(DATE_FORMAT) for day in days]
                    past_prices = stock_values
                    future_days = len(prediction)
                    predicted_dates = [(days[-1] + timedelta(days=i+1)).strftime(DATE_FORMAT) for i in range(future_days)]
                    
                    # Combine Dates and Prices for Plotting
                    all_dates = past_dates + predicted_dates
                    all_prices = past_prices + prediction
                    
                    # Calculate percentage change
                    if past_prices:
                        last_past_price = past_prices[-1]
                        up_percent = ((prediction[-1] - last_past_price) / last_past_price) * 100
                    else:
                        up_percent = 0.0
                    
                    # Visualization
                    st.subheader(f"Stock Price Prediction for {stock_code_upper}")
                    
                    fig = go.Figure()
                    
                    # Past Prices
                    fig.add_trace(go.Scatter(
                        x=past_dates,
                        y=past_prices,
                        mode='lines+markers',
                        name='Past Prices',
                        line=dict(color='#81A1C1', width=2),
                        marker=dict(size=6)
                    ))
                    
                    # Predicted Prices
                    fig.add_trace(go.Scatter(
                        x=predicted_dates,
                        y=prediction,
                        mode='lines+markers',
                        name='Predicted Prices',
                        line=dict(color='#88C0D0', width=3, dash='dot'),
                        marker=dict(size=8)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=[past_dates[-1], predicted_dates[0]],
                        y=[past_prices[-1], prediction[0]],
                        mode='lines+markers',
                        name='Predicted Prices',
                        line=dict(color='#88C0D0', width=3, dash='dot'),
                        marker=dict(size=8)
                    ))
                    
                    
                    # Layout Enhancements
                    fig.update_layout(
                        title=f"Stock Price Projection for {stock_code_upper}",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        hovermode='x unified',
                        template='plotly_dark',
                        legend=dict(x=0.01, y=0.99),
                        plot_bgcolor='#2E3440',
                        paper_bgcolor='#2E3440',
                        font=dict(color='white')
                    )
                    
                    # Add subtle animation to graph loading (Plotly inherently has animations)
                    
                    st.plotly_chart(fig, use_container_width=True, height=600)
                    
                    st.markdown("---")
                    st.subheader("Prediction Metrics")
                    col1 = st.columns(1)[0]
                    col1.metric("Percentage Change", f"{up_percent:.2f}%")
                    
                    st.markdown("### Additional Insights")
                    st.write("""
                    - **Past Prices:** Historical stock prices leading up to the prediction.
                    - **Predicted Prices:** Future stock prices as predicted by the model.
                    - **Connection Line:** Indicates the transition from past data to predicted data.
    
                    **Hover** over the chart to see detailed values.
                    """)
                    
                    # Disclaimer
                    st.markdown("""
                    ---
                    <div style='text-align: center; color: #BF616A;'>
                    <strong>Disclaimer:</strong> The predictions provided by this model are not guaranteed to be accurate. Use them at your own risk.
                    </div>
                    """, unsafe_allow_html=True)
    
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.error(traceback.format_exc())

if __name__ == "__main__":
    main()