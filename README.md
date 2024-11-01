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

    ```bash
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

    ```bash
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

    ```bash
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

