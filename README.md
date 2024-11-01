# Project Documentation

## Overview

This project contains a set of Python scripts to implement a stock prediction system. Each file serves a specific purpose, from data fetching and environment setup to model building and utility functions. Below is a breakdown of each file and the classes/functions they contain.

---

## Files and Their Purpose

### 1. `fetch_data.py`
This script is responsible for retrieving and preprocessing data for the stock prediction system.

- **Main Classes and Functions:**
  - `fetch_stock_data`: Retrieves historical stock data from a specified source (e.g., yfinance).
  - `fetch_macro_data`: Collects macroeconomic indicators, like GDP growth, interest rates, and unemployment, for model input.
  - `preprocess_data`: Cleans and organizes data for consistent input to the model.

### 2. `app.py`
This file serves as the main application interface. It provides a user interface to interact with the prediction system and view results.

- **Main Classes and Functions:**
  - `App`: Manages the application's main interface, handling user interactions and displaying predictions.
  - `run`: Initializes and runs the application.
  - `plot_results`: Displays the model's predictions and actual data on a graph for easy visualization.

### 3. `model.py`
Defines the main prediction model architecture, using a combination of layers and configurations to predict stock values based on past data and macroeconomic factors.

- **Main Classes and Functions:**
  - `StockPredictor`: The core model class, which combines LSTM and Transformer layers to predict future stock prices.
  - `forward`: Handles the forward pass of the model, taking in processed input and returning predictions.
  - `train`: A method to train the model on historical stock and macroeconomic data.

### 4. `stock_env.py`
Implements the environment in which the model interacts, especially relevant for reinforcement learning setups. This environment simulates stock market conditions and provides necessary state information to the model.

- **Main Classes and Functions:**
  - `StockEnv`: A class representing the stock environment, providing states, rewards, and steps for each simulation.
  - `reset`: Resets the environment to an initial state for a new simulation episode.
  - `step`: Advances the environment by one step, providing new data and rewards based on the model's actions.

### 5. `model_blocks.py`
Contains modular building blocks for the `StockPredictor` model. These blocks can be used independently or together to construct various architectures.

- **Main Classes and Functions:**
  - `AttentionBlock`: Implements attention mechanisms to focus on important time steps in the input sequence.
  - `LSTMBlock`: A wrapper around LSTM layers for ease of integration into larger model architectures.
  - `TransformerBlock`: A basic transformer encoder block used to capture dependencies in the input data.

### 6. `utility.py`
Provides helper functions and utilities used throughout the project for tasks like data transformation, evaluation, and metric calculations.

- **Main Classes and Functions:**
  - `normalize_data`: Standardizes data to have zero mean and unit variance.
  - `calculate_metrics`: Computes evaluation metrics such as MAE, MSE, and RMSE to assess model performance.
  - `save_model` and `load_model`: Functions to persist and load trained models, enabling model reuse.

---

## How to Use

1. **Set up environment**: Ensure all dependencies are installed.
2. **Run `app.py`**: Start the main application interface to interact with the prediction system.
3. **Fetch Data**: Use `fetch_data.py` to gather and preprocess required stock and macroeconomic data.
4. **Train the Model**: Use `model.py` to train the model on historical data.
5. **Environment Simulation**: If using reinforcement learning, use `stock_env.py` to simulate stock market conditions.

---

## Future Improvements

- Add more features to the environment for enhanced simulations.
- Experiment with additional model architectures in `model_blocks.py`.
- Enhance the user interface in `app.py` with more visualization options.

---

## Contributing

For contributing guidelines, please contact the author or submit a pull request with suggested changes.

---