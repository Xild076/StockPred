import os
import json
import pickle
import logging
import random
import datetime
import time
import warnings
from contextlib import nullcontext

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
from torch.utils.data import Dataset, DataLoader

from env import StockEnv
from econ_data import EconData
from fetch_fred import fetch_fred_data

# Suppress specific warnings
warnings.filterwarnings(
    "ignore",
    message=".*verbose parameter is deprecated.*",
    category=UserWarning,
    module="torch.optim.lr_scheduler"
)
warnings.filterwarnings(
    "ignore",
    message=".*You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
    module="torch"
)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
logging.info(f"Using device: {device}")

# Constants
STOCK_KEYS = [
    "AAPL", "MSFT", "GOOG", "V", "JNJ", "WMT", 
    "NVDA", "PG", "DIS", "MA", "HD", "VZ", "PFE", "PEP", "XOM", 
    "BAC", "MRK", "JPM", "GE", "C", "CVX", "ORCL", "IBM", "GILD"
]

HISTORY_FILE_NAME = 'models/history.json'
SCALER_FILE_NAME = 'models/scaler.pkl'
MODEL_DIR = 'models'

os.makedirs(MODEL_DIR, exist_ok=True)

# Load or initialize history
if os.path.exists(HISTORY_FILE_NAME):
    try:
        with open(HISTORY_FILE_NAME, 'r') as json_file:
            history = json.load(json_file)
    except json.JSONDecodeError:
        logging.warning(f"{HISTORY_FILE_NAME} is empty or invalid. Initializing with empty history.")
        history = {}
else:
    logging.info(f"{HISTORY_FILE_NAME} does not exist. Creating a new one.")
    history = {}

def update_history(model_name, model_data):
    history[model_name] = model_data
    with open(HISTORY_FILE_NAME, 'w') as json_file:
        json.dump(history, json_file, indent=4)
    logging.info(f"History updated for model: {model_name}")

def read_history(model_name):
    return history.get(model_name, None)

# Attention Mechanism
class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super(Attention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.layer_norm(x + attn_output)
        return x

# LSTM Model with Combined Output
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, 
                 num_layers, output_size, attention=True, 
                 dropout=0.5, bidirectional=True):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3 if num_layers > 1 else 0
        )

        self.attention = Attention(hidden_size * self.num_directions) if attention else None

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, self.output_size)
        )

        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        if self.attention:
            out = self.attention(out)
            out = torch.mean(out, dim=1)
        else:
            out = out[:, -1, :]
        output = self.fc(out)
        return output

# Transformer Model with Combined Output
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_heads=4, dropout=0.3):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_size = output_size
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size // 2, self.output_size)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.layer_norm(x)
        x = torch.mean(x, dim=1)
        output = self.fc(x)
        return output

# Unified Model Class
class Model(nn.Module):
    def __init__(self, model_type, input_size, hidden_size, num_layers, output_size, num_heads=4, attention=True, dropout=0.5, bidirectional=True):
        super(Model, self).__init__()
        if model_type.lower() == 'lstm':
            self.model = LSTMModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size,
                attention=attention,
                dropout=dropout,
                bidirectional=bidirectional
            )
        elif model_type.lower() == 'transformer':
            self.model = TransformerModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            raise ValueError("model_type must be 'lstm' or 'transformer'")

    def forward(self, x):
        return self.model(x)

# Custom Dataset
class StockDataset(Dataset):
    def __init__(self, data, stock_list, features, scaler, add_noise_func, input_days, predict_days):
        self.data = data
        self.stock_list = stock_list
        self.features = features
        self.scaler = scaler
        self.add_noise = add_noise_func
        self.input_days = input_days
        self.predict_days = predict_days

    def __len__(self):
        return len(self.data) * len(self.stock_list)

    def __getitem__(self, idx):
        sample_idx = idx // len(self.stock_list)
        stock_idx = idx % len(self.stock_list)
        data, labels = self.data[sample_idx]
        stock = self.stock_list[stock_idx]
        stock_data = data[stock]
        regression_target = labels[stock][:self.predict_days]
        previous_value = stock_data[-1]['Stock Value']
        
        feature_values = [
            float(item.get(feature, 0.0)) if self.is_number(item.get(feature, 0)) else 0.0
            for item in stock_data[-self.input_days:]
            for feature in self.features
        ]
        feature_values = np.array(feature_values).reshape(-1, len(self.features))
        scaled_data = self.scaler.transform(feature_values)
        tensor_data = torch.tensor(scaled_data, dtype=torch.float32)
        tensor_data = self.add_noise(tensor_data)
        regression_target = torch.tensor(regression_target, dtype=torch.float32)
        previous_value = torch.tensor(previous_value, dtype=torch.float32)
        direction_target = (regression_target > previous_value).float()
        return tensor_data, regression_target, direction_target

    @staticmethod
    def is_number(value):
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

# Stock Predictor Class
class StockPredictor:
    def __init__(
        self,
        model_type='transformer',
        stock_list=STOCK_KEYS,
        learning_rate=0.001,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        scaler=MinMaxScaler,
        attention=True,
        dropout=0.3,
        bidirectional=True,
        model_name=None,
        use_tqdm=True,
        input_days=10,
        predict_days=2,
        early_stopping_patience=10
    ):
        self.features = [
            'Stock Value', 'Volume', 'Net Income', 'EPS', 'PE Ratio',
            'Dividend Yield', 'Market Cap', 'Beta', 'PB Ratio', 'ROE',
            'Revenue', 'Gross Profit', 'EBITDA', 'Operating Income',
            'Total Assets', 'Total Liabilities', 'Total Debt', 'Cash and Cash Equivalents',
            'Current Ratio', 'Debt to Equity Ratio', 'Profit Margin', 'Return on Assets',
            'MA50', 'MA200', 'RSI', 'MACD', 'ADX',
            'T10YIE', 'DFF', 'DGS10', 'DEXUSEU',
            'UNRATE', 'GFDEGDQ188S', 'A191RL1Q225SBEA', 'M2SL', 'CPIAUCSL', 'UMCSENT',
            'BSCICP03USM665S', 'GS10', 'INDPRO', 'PAYEMS', 'PCE', 'RSAFS', 'CPATAX',
            'HOUST', 'Sentiment', 'Current Assets', 'Current Liabilities',
            'Operating Cash Flow', 'Free Cash Flow'
        ]

        self.input_days = input_days
        self.predict_days = predict_days
        self.input_size = len(self.features)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.stock_list = stock_list if stock_list else []
        self.dropout = dropout
        self.attention = attention
        self.model_type = model_type.lower()

        self.model_name = model_name if model_name else self._make_model_name()

        self.model = Model(
            model_type=self.model_type,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.predict_days,
            num_heads=self.num_heads,
            attention=self.attention,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        ).to(device)

        self.lr = learning_rate
        self.criterion_regression = nn.SmoothL1Loss()
        self.criterion_direction = nn.BCEWithLogitsLoss()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True
        )

        self.scaler = scaler()
        self.model_data = None

        self.use_amp = device.type in ['cuda']
        self.scaler_amp = amp.GradScaler() if self.use_amp else None

        self.use_tqdm = use_tqdm

        # Initialize environment (Assuming StockEnv and EconData are defined elsewhere)
        self.date_range = ['2018-01-01', '2024-08-01']
        self.env = StockEnv(self.stock_list, self.date_range)

        self.loss_values = []
        self.val_loss_values = []
        self.epochs = []
        self.accuracy = []

        if model_name:
            self.load_model()

        # Early Stopping parameters
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

    def _make_model_name(self):
        base_name = f'{self.model_type.capitalize()}'
        base_name += f'_Heads{self.num_heads}' if self.model_type == 'transformer' else ''
        base_name += f'_Layers{self.num_layers}'
        base_name += f'_Hidden{self.hidden_size}'
        base_name += f'_Input{self.input_days}_Predict{self.predict_days}'
        base_name += f'_{datetime.datetime.now().strftime("%Y%m%d%H%M")}'
        return base_name

    def _generate_nickname(self, existing_nicknames):
        numbers = list('0123456789')
        letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        celestial_objects = [
            "Mercury", "Venus", "Earth", "Mars", "Jupiter",
            "Saturn", "Uranus", "Neptune", "Orion", "Andromeda",
            "Pegasus", "Cassiopeia", "Lyra", "Cygnus", "Draco",
            "Phoenix", "Hydra", "Scorpius", "Taurus", "Gemini",
            "Perseus", "Sagittarius", "Capricornus", "Aquarius", "Pisces",
            "Leo", "Virgo", "Cancer", "Libra", "Gemini",
            "Crux", "Corona", "Hercules", "Monoceros", "Canis Major",
            "Canis Minor", "Cetus", "Lepus", "Auriga", "Ophiuchus"
        ]
        while True:
            number = random.choice(numbers)
            letter = random.choice(letters)
            celestial_object = random.choice(celestial_objects)
            nickname = f"{number}{letter} {celestial_object}"
            if nickname not in existing_nicknames:
                return nickname

    @staticmethod
    def is_number(value):
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def add_noise(self, tensor):
        noise_level = 0.01
        return tensor + torch.randn_like(tensor) * noise_level

    def fit_scaler(self):
        if not self.stock_list:
            logging.error("stock_list is empty. Please provide at least one stock symbol.")
            raise ValueError("stock_list is empty. Please provide at least one stock symbol.")
        
        all_feature_data = []
        logging.info("Fitting scaler with collected feature data...")
        for _ in tqdm(range(100), desc="Fitting Scaler", disable=not self.use_tqdm):
            data, labels, _ = self.env.calc_state(self.input_days, self.predict_days)
            for stock in self.stock_list:
                for item in data[stock][-self.input_days:]:
                    feature_values = [
                        float(item.get(feature, 0.0)) if self.is_number(item.get(feature, 0)) else 0.0
                        for feature in self.features
                    ]
                    all_feature_data.append(feature_values)
        
        if not all_feature_data:
            logging.error("No feature data collected. Ensure that data is available for the provided stock_list.")
            raise ValueError("No feature data collected. Ensure that data is available for the provided stock_list.")
        
        self.scaler.fit(all_feature_data)
        with open(SCALER_FILE_NAME, 'wb') as f:
            pickle.dump(self.scaler, f)
        logging.info(f"Scaler fitted and saved to {SCALER_FILE_NAME}")

    def load_scaler(self):
        if os.path.exists(SCALER_FILE_NAME):
            with open(SCALER_FILE_NAME, 'rb') as f:
                self.scaler = pickle.load(f)
            logging.info(f"Scaler loaded from {SCALER_FILE_NAME}")
        else:
            logging.warning(f"Scaler file {SCALER_FILE_NAME} not found. Fitting scaler.")
            self.fit_scaler()

    def custom_loss(self, regression_outputs, regression_targets, direction_targets, mse_weight, dir_weight):
        mse_loss = self.criterion_regression(regression_outputs, regression_targets) * mse_weight

        previous_values = regression_targets[:, 0].unsqueeze(1) 
        direction_logits = regression_outputs - previous_values 

        direction_loss = self.criterion_direction(direction_logits, direction_targets) * dir_weight

        return mse_loss + direction_loss

    def train_model(self, num_epochs, batch_size, validation_split=0.1, accumulate_steps=1, mse_weight=1.0, dir_weight=1.0):
        self.model.train()
        self.load_scaler()

        all_data = []
        total_iterations = batch_size * num_epochs * 2
        logging.info(f"Collecting training data: {total_iterations} iterations")
        for _ in tqdm(range(total_iterations), desc="Collecting Training Data", disable=not self.use_tqdm):
            data, labels, _ = self.env.calc_state(self.input_days, self.predict_days)
            all_data.append((data, labels))
        
        split_idx = int(len(all_data) * (1 - validation_split))
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]

        if device.type == 'mps':
            num_workers = 0
            pin_memory = False
        else:
            num_workers = 4
            pin_memory = self.use_amp 
        
        train_dataset = StockDataset(train_data, self.stock_list, self.features, self.scaler, self.add_noise, self.input_days, self.predict_days)
        val_dataset = StockDataset(val_data, self.stock_list, self.features, self.scaler, self.add_noise, self.input_days, self.predict_days)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )

        for epoch in range(num_epochs):
            start_time = time.time()
            self.model.train()
            total_loss = 0.0

            for i, (inputs_batch, regression_targets, direction_targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not self.use_tqdm)):
                inputs_batch = inputs_batch.to(device)
                regression_targets = regression_targets.to(device)
                direction_targets = direction_targets.to(device)

                self.optimizer.zero_grad()

                if self.use_amp and device.type == 'cuda':
                    with amp.autocast(device_type='cuda', enabled=True):
                        regression_outputs = self.model(inputs_batch)
                        loss = self.custom_loss(regression_outputs, regression_targets, direction_targets, mse_weight, dir_weight)
                        loss = loss / accumulate_steps
                else:
                    with nullcontext():
                        regression_outputs = self.model(inputs_batch)
                        loss = self.custom_loss(regression_outputs, regression_targets, direction_targets, mse_weight, dir_weight)
                        loss = loss / accumulate_steps

                if self.use_amp and device.type == 'cuda':
                    self.scaler_amp.scale(loss).backward()
                else:
                    loss.backward()

                if (i + 1) % accumulate_steps == 0:
                    if self.use_amp and device.type == 'cuda':
                        self.scaler_amp.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler_amp.step(self.optimizer)
                        self.scaler_amp.update()
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()

                total_loss += loss.item() * accumulate_steps

            avg_train_loss = total_loss / len(train_loader)
            self.loss_values.append(avg_train_loss)

            # Validation
            val_loss = self.evaluate(val_loader, mse_weight, dir_weight)
            self.val_loss_values.append(val_loss)

            accuracy, _, _ = self.test_accuracy()
            self.accuracy.append(accuracy)

            # Scheduler step
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']


            self.epochs.append(epoch + 1)

            # Logging
            logging.info(
                f'Epoch {epoch+1}/{num_epochs}, '
                f'Train Loss: {avg_train_loss:.6f}, '
                f'Val Loss: {val_loss:.6f}, '
                f'LR: {current_lr:.6f}, '
                f'Time: {time.time() - start_time:.2f}s'
            )

            # Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.save_model()
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.early_stopping_patience:
                    logging.info("Early stopping triggered.")
                    break

    def evaluate(self, val_loader, mse_weight, dir_weight):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs_batch, regression_targets, direction_targets in tqdm(val_loader, desc="Evaluating", disable=not self.use_tqdm):
                inputs_batch = inputs_batch.to(device)
                regression_targets = regression_targets.to(device)
                direction_targets = direction_targets.to(device)

                if self.use_amp and device.type == 'cuda':
                    with amp.autocast(device_type='cuda', enabled=True):
                        regression_outputs = self.model(inputs_batch)
                        loss = self.custom_loss(regression_outputs, regression_targets, direction_targets, mse_weight, dir_weight)
                else:
                    with nullcontext():
                        regression_outputs = self.model(inputs_batch)
                        loss = self.custom_loss(regression_outputs, regression_targets, direction_targets, mse_weight, dir_weight)

                total_loss += loss.item()

        avg_val_loss = total_loss / len(val_loader)
        logging.info(f'Validation Loss: {avg_val_loss:.6f}')
        return avg_val_loss

    def save_model(self):
        model_path = os.path.join(MODEL_DIR, f"{self.model_name}.pt")
        model_data = {
            'nickname': self.model_data.get('nickname') if self.model_data else self._generate_nickname(),
            'model_type': self.model_type,
            'epoch': self.epochs[-1],
            'train_loss': self.loss_values[-1],
            'val_loss': self.val_loss_values[-1],
            'accuracy': self.accuracy[-1],
            'lr': self.optimizer.param_groups[0]['lr'],
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'attention': self.attention,
            'input_days': self.input_days,
            'predict_days': self.predict_days,
            'loss_weights': {'mse': 1.0, 'dir': 1.0}
        }
        update_history(self.model_name, model_data)
        with open(SCALER_FILE_NAME, 'wb') as f:
            pickle.dump(self.scaler, f)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'model_data': model_data,
        }, model_path)
        logging.info(f"Model saved to {model_path}")

    def load_model(self):
        model_path = os.path.join(MODEL_DIR, f"{self.model_name}.pt")
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device)
                # Load state_dict with strict=False to allow mismatches
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.load_scaler()
                self.model_data = checkpoint.get('model_data', read_history(self.model_name))
                logging.info(f"Model loaded from {model_path}")
            except RuntimeError as e:
                logging.error(f"RuntimeError while loading the model: {e}")
                logging.error("Attempting to load with strict=False...")
                # Attempt to load with strict=False
                try:
                    self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    self.load_scaler()
                    self.model_data = checkpoint.get('model_data', read_history(self.model_name))
                    logging.info(f"Model loaded with partial state_dict from {model_path}")
                except Exception as ex:
                    logging.error(f"Failed to load model with strict=False: {ex}")
                    raise ex
        else:
            logging.error(f"No model found at {model_path}")

    def test_accuracy(self, test_data_size=100):
        self.model.eval()
        correct_directions = 0
        total = 0
        mae = 0.0
        rmse = 0.0
        with torch.no_grad():
            for _ in tqdm(range(test_data_size), desc="Testing Accuracy", disable=not self.use_tqdm):
                data, labels, _ = self.env.calc_state(self.input_days, self.predict_days)
                for stock in self.stock_list:
                    stock_data = data[stock]
                    feature_values = [
                        float(item.get(feature, 0.0)) if self.is_number(item.get(feature, 0)) else 0.0
                        for item in stock_data[-self.input_days:]
                        for feature in self.features
                    ]
                    feature_values = np.array(feature_values).reshape(-1, len(self.features))
                    scaled_data = self.scaler.transform(feature_values)
                    tensor_data = torch.tensor(scaled_data, dtype=torch.float32).to(device)
                    tensor_data = self.add_noise(tensor_data).unsqueeze(0)
                    regression_output = self.model(tensor_data).squeeze(0).cpu().numpy()
                    
                    previous_value = stock_data[-1]['Stock Value']
                    actual_values = labels[stock][:self.predict_days]
                    
                    # Determine direction from regression output
                    predicted_directions = (regression_output > previous_value).astype(float)
                    actual_directions = (actual_values > previous_value).astype(float)
                    
                    correct_directions += (predicted_directions == actual_directions).sum()
                    total += len(predicted_directions)
                    mae += np.mean(np.abs(regression_output - actual_values))
                    rmse += np.sqrt(np.mean((regression_output - actual_values) ** 2))
        accuracy = (correct_directions / total) * 100 if total > 0 else 0.0
        mae /= (test_data_size * len(self.stock_list)) if (test_data_size * len(self.stock_list)) > 0 else 1
        rmse /= (test_data_size * len(self.stock_list)) if (test_data_size * len(self.stock_list)) > 0 else 1
        logging.info(f'Test Accuracy (Direction Match): {accuracy:.2f}%')
        logging.info(f'Test MAE: {mae:.2f}')
        logging.info(f'Test RMSE: {rmse:.2f}')
        return accuracy, mae, rmse

    def predict(self, data_dict, stock_symbol):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.prepare_model_input(data_dict, stock_symbol))
        return output

    def prepare_model_input(self, data_dict, stock_symbol):
        if stock_symbol not in data_dict:
            logging.error(f"Stock symbol '{stock_symbol}' not found in the provided data.")
            raise ValueError(f"Stock symbol '{stock_symbol}' not found in the provided data.")

        stock_data = data_dict[stock_symbol]

        if len(stock_data) < self.input_days:
            logging.error(
                f"Not enough data for stock '{stock_symbol}'. "
                f"Required: {self.input_days}, Available: {len(stock_data)}."
            )
            raise ValueError(
                f"Not enough data for stock '{stock_symbol}'. "
                f"Required: {self.input_days}, Available: {len(stock_data)}."
            )

        relevant_data = stock_data[-self.input_days:]

        feature_values = []
        for day_data in relevant_data:
            day_features = []
            for feature in self.features:
                value = day_data.get(feature, day_data.get(feature.lower(), 0.0))
                if self.is_number(value):
                    day_features.append(float(value))
                else:
                    day_features.append(0.0)
            feature_values.append(day_features)

        feature_array = np.array(feature_values)
        scaled_data = self.scaler.transform(feature_array)
        tensor_data = torch.tensor(scaled_data, dtype=torch.float32)
        tensor_data = tensor_data.unsqueeze(0)

        return tensor_data.to(device)

    def visualize_predictions(self, stock_symbol):
        self.model.eval()
        data, labels, dates = self.env.calc_state(self.input_days, self.predict_days, specific_stock=stock_symbol)
        stock_data = data[stock_symbol]
        feature_values = [
            float(item.get(feature, 0.0)) if self.is_number(item.get(feature, 0)) else 0.0
            for item in stock_data[-self.input_days:]
            for feature in self.features
        ]
        feature_values = np.array(feature_values).reshape(-1, len(self.features))
        scaled_data = self.scaler.transform(feature_values)
        tensor_data = torch.tensor(scaled_data, dtype=torch.float32).to(device)
        tensor_data = self.add_noise(tensor_data).unsqueeze(0)
        with torch.no_grad():
            regression_output = self.model(tensor_data).squeeze(0).cpu().numpy()
        future_prices = []
        future_dates = []
        for i in range(self.predict_days):
            if self.input_days + i >= len(dates):
                break
            date_str = (dates[self.input_days + i]).strftime("%Y-%m-%d")
            stock_data_fut = EconData.get_company_info(stock_symbol, date_str, fetch_fred_data(date_str))
            if stock_data_fut and 'Stock Value' in stock_data_fut:
                future_prices.append(stock_data_fut['Stock Value'])
                future_dates.append(date_str)
        past_prices = [item['Stock Value'] for item in stock_data[-self.input_days:]]
        past_dates = [item['date'] for item in stock_data[-self.input_days:]]
        min_length = min(len(regression_output), len(future_prices))
        predicted_prices = regression_output[:min_length]
        future_prices = future_prices[:min_length]
        future_dates = future_dates[:min_length]
        last_past_date = past_dates[-1]
        last_past_price = past_prices[-1]
        predicted_dates = [last_past_date] + future_dates
        predicted_prices_plot = [last_past_price] + list(predicted_prices)
        actual_prices_plot = [last_past_price] + future_prices
        plt.figure(figsize=(12, 6))
        plt.plot(past_dates, past_prices, label='Past Prices', marker='o', markersize=6)
        plt.plot(predicted_dates, predicted_prices_plot, label='Predicted Prices', marker='o', linestyle='--', color='orange')
        plt.plot(predicted_dates, actual_prices_plot, label='Actual Future Prices', marker='o', linestyle='-', color='green')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title(f'Stock Price Prediction for {stock_symbol}')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def _create_nickname(self, existing_nicknames):
        numbers = list('0123456789')
        letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        celestial_objects = [
            "Mercury", "Venus", "Earth", "Mars", "Jupiter",
            "Saturn", "Uranus", "Neptune", "Orion", "Andromeda",
            "Pegasus", "Cassiopeia", "Lyra", "Cygnus", "Draco",
            "Phoenix", "Hydra", "Scorpius", "Taurus", "Gemini",
            "Perseus", "Sagittarius", "Capricornus", "Aquarius", "Pisces",
            "Leo", "Virgo", "Cancer", "Libra", "Gemini",
            "Crux", "Corona", "Hercules", "Monoceros", "Canis Major",
            "Canis Minor", "Cetus", "Lepus", "Auriga", "Ophiuchus"
        ]
        
        max_attempts = 1000 
        attempts = 0
        
        while attempts < max_attempts:
            number = random.choice(numbers)
            letter = random.choice(letters)
            celestial_object = random.choice(celestial_objects)
            nickname = f"{number}{letter} {celestial_object}"
            if nickname not in existing_nicknames:
                return nickname
            attempts += 1
        
        raise ValueError("Unable to generate a unique nickname after multiple attempts.")

    def _generate_nickname(self):
        existing_nicknames = [data.get('nickname', '') for data in history.values()]
        return self._create_nickname(existing_nicknames)

