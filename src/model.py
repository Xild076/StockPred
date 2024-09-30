import torch
import torch.nn as nn
import torch.optim as optim
from env import StockEnv
from econ_data import EconData
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
from fetch_fred import fetch_fred_data
import datetime
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gc
from torch import amp
import json
import pickle
import logging
import random

import warnings

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stock_keys = [
    "AAPL", "MSFT", "GOOG", "V", "JNJ", "WMT", 
    "NVDA", "PG", "DIS", "MA", "HD", "VZ", "PFE", "PEP", "XOM", 
    "BAC", "MRK", "JPM", "GE", "C", "CVX", "ORCL", "IBM", "GILD"
]

HISTORY_FILE_NAME = 'models/history.json'
SCALER_FILE_NAME = 'models/scaler.pkl'

with open(HISTORY_FILE_NAME, 'r') as json_file:
    history = json.load(json_file)

def update_history(model_name, model_data):
    data = {}
    if os.path.exists(HISTORY_FILE_NAME):
        try:
            with open(HISTORY_FILE_NAME, 'r') as json_file:
                data = json.load(json_file)
        except json.JSONDecodeError:
            logging.warning(f"{HISTORY_FILE_NAME} is empty or invalid. Initializing with empty history.")
    else:
        logging.info(f"{HISTORY_FILE_NAME} does not exist. Creating a new one.")
    
    data[model_name] = model_data
    
    with open(HISTORY_FILE_NAME, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    logging.info(f"History updated for model: {model_name}")

def read_history(model_name):
    if not os.path.exists(HISTORY_FILE_NAME):
        logging.warning(f"{HISTORY_FILE_NAME} does not exist.")
        return None
    try:
        with open(HISTORY_FILE_NAME, 'r') as json_file:
            data = json.load(json_file)
        return data.get(model_name, None)
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from {HISTORY_FILE_NAME}.")
        return None

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super(Attention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.layer_norm(x + attn_output)
        return x

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

        self.fc_regression = nn.Sequential(
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
        regression_output = self.fc_regression(out)
        return regression_output

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
        self.fc_regression = nn.Sequential(
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
        regression_output = self.fc_regression(x)
        return regression_output

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
            raise ValueError("Model_type must be 'lstm' or 'transformer'")

    def forward(self, x):
        return self.model(x)

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
        return tensor_data, regression_target, previous_value

    def is_number(self, value):
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

class StockPredictor:
    def __init__(
        self,
        model_type='transformer',
        stock_list=stock_keys,
        learning_rate=0.001,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        scaler=MinMaxScaler,
        attention=True,
        dropout=0.5,
        bidirectional=True,
        model_name=None,
        use_tqdm=True,
        input_days=10,
        predict_days=2
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
        self.criterion_mae = nn.L1Loss()
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        self.scaler = scaler()
        os.makedirs('models', exist_ok=True)

        self.model_data = None

        self.use_amp = device.type == 'cuda'
        self.scaler_amp = amp.GradScaler() if self.use_amp else None

        self.use_tqdm = use_tqdm

        self.date_range = ['2018-01-01', '2024-08-01']
        self.env = StockEnv(self.stock_list, self.date_range)

        self.loss_values = []
        self.val_loss_values = []
        self.epochs = []

        if model_name:
            self.load_model()
            self.model_data = read_history(model_name)

        # self.fit_scalar()

    def _make_model_name(self):
        base_name = f'{self.model_type.capitalize()}'
        base_name += f'_Heads{self.num_heads}' if self.model_type == 'transformer' else ''
        base_name += f'_Layers{self.num_layers}'
        base_name += f'_Hidden{self.hidden_size}'
        base_name += f'_Input{self.input_days}_Predict{self.predict_days}'
        base_name += f'_{datetime.datetime.now().strftime("%Y%m%d%H%M")}'
        return base_name

    def _generate_nickname(self, existing_nicknames):
        adjectives = [
            "5p", "6p", "9v", "7c", "9x",
            "9j", "6n", "5f", "5c", "4d",
            "3e", "8t", "7n", "0m", "1n"
        ]
        nouns = [
            "Mercury", "Venus", "Earth", "Mars", "Jupiter",
            "Saturn", "Uranus", "Neptune"
        ]
        while True:
            adjective = random.choice(adjectives)
            noun = random.choice(nouns)
            nickname = f"{adjective} {noun}"
            if nickname not in existing_nicknames:
                return nickname

    def is_number(self, value):
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def add_noise(self, tensor):
        noise_level = 0.01
        return tensor + torch.randn_like(tensor) * noise_level

    def fit_scalar(self):
        if not self.stock_list:
            logging.error("stock_list is empty. Please provide at least one stock symbol.")
            raise ValueError("stock_list is empty. Please provide at least one stock symbol.")
        
        all_feature_data = []
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

    def custom_loss(self, regression_outputs, regression_targets, previous_values, mse_weight, mae_weight, dir_weight):
        previous_values = previous_values.unsqueeze(1).expand_as(regression_targets)
        mse_loss = self.criterion_regression(regression_outputs, regression_targets) * mse_weight
        mae_loss = self.criterion_mae(regression_outputs, regression_targets) * mae_weight
        delta_b = regression_outputs - previous_values
        delta_c = regression_targets - previous_values
        dir_loss = torch.relu(-delta_b * delta_c).mean() * dir_weight
        return mse_loss + mae_loss + dir_loss

    def train(self, num_epochs, batch_size, validation_split=0.1, accumulate_steps=1, mse_weight=1.0, mae_weight=1.0, dir_weight=5.0):
        self.model.train()
        if os.path.exists(SCALER_FILE_NAME):
            with open(SCALER_FILE_NAME, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            self.fit_scalar()
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
            for i, (inputs_batch, regression_targets, previous_values) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not self.use_tqdm)):
                inputs_batch = inputs_batch.to(device)
                regression_targets = regression_targets.to(device)
                previous_values = previous_values.to(device)
                if self.use_amp:
                    with amp.autocast():
                        regression_outputs = self.model(inputs_batch)
                        loss = self.custom_loss(regression_outputs, regression_targets, previous_values, mse_weight, mae_weight, dir_weight)
                        loss = loss / accumulate_steps
                    self.scaler_amp.scale(loss).backward()
                else:
                    regression_outputs = self.model(inputs_batch)
                    loss = self.custom_loss(regression_outputs, regression_targets, previous_values, mse_weight, mae_weight, dir_weight)
                    loss = loss / accumulate_steps
                    loss.backward()

                if (i + 1) % accumulate_steps == 0:
                    if self.use_amp:
                        self.scaler_amp.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler_amp.step(self.optimizer)
                        self.scaler_amp.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item() * accumulate_steps

                del inputs_batch, regression_targets, previous_values, regression_outputs, loss
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif device.type == 'mps':
                    torch.mps.empty_cache()

            avg_train_loss = total_loss / len(train_loader)
            self.loss_values.append(avg_train_loss)
            val_loss = self.evaluate(val_loader, mse_weight, mae_weight, dir_weight)
            self.val_loss_values.append(val_loss)
            self.scheduler.step(val_loss)
            self.epochs.append(epoch + 1)
            current_lr = self.optimizer.param_groups[0]['lr']
            accuracy = self.test_accuracy()
            logging.info(
                f'Epoch {epoch+1}/{num_epochs}, '
                f'Train Loss: {avg_train_loss:.6f}, '
                f'Val Loss: {val_loss:.6f}, '
                f'LR: {current_lr:.6f}, '
                f'Time: {time.time() - start_time:.2f}s'
            )
            if self.model_data:
                self.model_data['epoch'] += 1
                self.model_data['train_loss'] = min(self.model_data.get('train_loss', float('inf')), avg_train_loss)
                self.model_data['val_loss'] = min(self.model_data.get('val_loss', float('inf')), val_loss)
                self.model_data['lr'] = current_lr
                self.model_data['accuracy'] = max(self.model_data.get('accuracy', 0.0), accuracy)
                self.model_data['loss_weights'] = {'mse': mse_weight, 'mae': mae_weight, 'dir': dir_weight}
                if not 'nickname' in list(self.model_data):
                    existing_nicknames = [data.get('nickname', '') for data in history.values()]
                    nickname = self._generate_nickname(existing_nicknames)
                    self.model_data['nickname'] = nickname
                self.model_data['hidden_size'] = self.hidden_size
                self.model_data['model_type'] = self.model_type
                self.model_data['num_layers'] = self.num_layers
                self.model_data['hidden_size'] = self.hidden_size
                self.model_data['num_heads'] = self.num_heads
                self.model_data['dropout'] = self.dropout
                self.model_data['bidirectional'] = self.bidirectional
                self.model_data['attention'] = self.attention
                self.model_data['predict_days'] = self.predict_days
                self.model_data['input_days'] = self.input_days
            else:
                existing_nicknames = [data.get('nickname', '') for data in history.values()]
                nickname = self._generate_nickname(existing_nicknames)
                
                self.model_data = {
                    'nickname': nickname,
                    'model_type': self.model_type,
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'lr': current_lr,
                    'accuracy': accuracy,
                    'hidden_size': self.hidden_size,
                    'num_layers': self.num_layers,
                    'num_heads': self.num_heads,
                    'dropout': self.dropout,
                    'bidirectional': self.bidirectional,
                    'attention': self.attention,
                    'input_days': self.input_days,
                    'predict_days': self.predict_days,
                }
            self.save_model(self.model_data)
        accuracy = self.test_accuracy()
        logging.info(f'Final Test Accuracy: {accuracy:.2f}%')
    
    def evaluate(self, val_loader, mse_weight, mae_weight, dir_weight):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs_batch, regression_targets, previous_values in tqdm(val_loader, desc="Evaluating", disable=not self.use_tqdm):
                inputs_batch = inputs_batch.to(device)
                regression_targets = regression_targets.to(device)
                previous_values = previous_values.to(device)
                if self.use_amp:
                    with amp.autocast():
                        regression_outputs = self.model(inputs_batch)
                        loss = self.custom_loss(regression_outputs, regression_targets, previous_values, mse_weight, mae_weight, dir_weight)
                else:
                    regression_outputs = self.model(inputs_batch)
                    loss = self.custom_loss(regression_outputs, regression_targets, previous_values, mse_weight, mae_weight, dir_weight)
                total_loss += loss.item()
                del inputs_batch, regression_targets, previous_values, regression_outputs, loss
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif device.type == 'mps':
                    torch.mps.empty_cache()
        avg_val_loss = total_loss / len(val_loader)
        logging.info(f'Validation Loss: {avg_val_loss:.6f}')
        return avg_val_loss

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
        return accuracy

    def save_model(self, model_data):
        model_path = f"models/{self.model_name}.pt"
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
        model_path = f"models/{self.model_name}.pt"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if os.path.exists(SCALER_FILE_NAME):
                with open(SCALER_FILE_NAME, 'rb') as f:
                    self.scaler = pickle.load(f)
                logging.info(f"Scaler loaded from {SCALER_FILE_NAME}")
            else:
                logging.warning(f"Scaler file {SCALER_FILE_NAME} not found. Refitting scaler.")
                self.fit_scalar()
            self.model_data = checkpoint.get('model_data', read_history(self.model_name))
            logging.info(f"Model loaded from {model_path}")
        else:
            logging.error(f"No model found at {model_path}")

    def _test_model_internal(self, stock_symbol):
        self.model.eval()
        input_days = self.input_days
        predict_days = self.predict_days
        data, _, dates = self.env.calc_state(input_days, predict_days, specific_stock=stock_symbol)
        stock_data = data[stock_symbol]
        feature_values = [
            float(item.get(feature, 0.0)) if self.is_number(item.get(feature, 0)) else 0.0
            for item in stock_data[-input_days:]
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
        for i in range(predict_days):
            if input_days + i >= len(dates):
                break
            date_str = (dates[input_days + i]).strftime("%Y-%m-%d")
            stock_data_fut = EconData.get_company_info(stock_symbol, date_str, fetch_fred_data(date_str))
            if stock_data_fut and 'Stock Value' in stock_data_fut:
                future_prices.append(stock_data_fut['Stock Value'])
                future_dates.append(date_str)
        past_prices = [item['Stock Value'] for item in stock_data[-input_days:]]
        past_dates = [item['date'] for item in stock_data[-input_days:]]
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

        return tensor_data

"""
model = StockPredictor(
        model_type='lstm',
        stock_list=['AAPL'],
        learning_rate=0.001,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        scaler=MinMaxScaler,
        attention=True,
        dropout=0.5,
        bidirectional=False,
        use_tqdm=True,
        model_name='Lstm_Layers2_Hidden128_Input15_Predict3_202409291535',
        input_days=15,
        predict_days=3,
)

if __name__ == '__main__':
    model.train(1, 64)"""