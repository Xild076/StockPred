import os
import pickle
import datetime
from datetime import datetime, timedelta
import json
from typing import Dict, Any, List, Optional, Literal

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from stock_env import StockDataset, ticker_symbols
from model_blocks import MODEL_BLOCKS
from utility import Logging

from torch.amp import autocast, GradScaler
from fetch_data import FetchStock, FetchFred, FetchSentiment
import time

device = torch.device("cpu")
log = Logging('log.txt')

SCALER_FILE_NAME = 'models/scaler.pkl'
MODEL_STATS_FILE_NAME = 'models/model_stats.json'

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse = self.mse_loss(outputs, targets)
        return mse

class StockPredictor:
    def __init__(
        self,
        model_type: str,
        stock_list: List[str],
        input_days: int,
        output_days: int,
        model_name: Optional[str] = None,
        lr: float = 0.001,
        hidden_size: int = 128,
    ):
        self.model_types = MODEL_BLOCKS
        self.model_type = model_type.lower() if model_type.lower() in self.model_types else 'lstm'
        self.stock_list = stock_list
        self.input_days = input_days
        self.output_days = output_days
        self.hidden_size = hidden_size
        self.lr = lr
        self.scaler = StandardScaler()
        self.label_scaler = StandardScaler()
        self.features = StockDataset.FEATURES
        self.date_range = StockDataset.DATE_RANGE
        self.dataset = StockDataset(
            self.stock_list,
            self.date_range,
            self.input_days,
            self.output_days,
            technical_indicators=['MACD', 'RSI']
        )
        self.load_scaler()
        self.build_model()
        self.train_losses = []
        self.val_losses = []
        self.epoch = 0
        if model_name:
            self.model_name = model_name
            self.load_model()
        else:
            self.model_name = self._generate_model_name()
            loaded_stats = self.load_model_stats()
            existing_nicks = [loaded_stats[model_n]['model_nickname'] for model_n in loaded_stats]
            self._model_nickname = self._generate_model_nickname(existing_nicknames=existing_nicks)

    def load_scaler(self):
        if os.path.exists(SCALER_FILE_NAME):
            with open(SCALER_FILE_NAME, 'rb') as f:
                self.scaler, self.label_scaler = pickle.load(f)
            log.success('Scaler loaded!')
        else:
            self.fit_scaler()
            log.alert('Scaler not found. A new scaler has been fitted.')

    def fit_scaler(self):
        log.log('Fitting scaler...')
        all_feature_data = []
        all_label_data = []
        for i in tqdm(range(len(self.dataset)), desc="Fitting Scaler"):
            data, label = self.dataset[i]
            features = data.numpy()
            features = features.reshape(-1, features.shape[-1])
            all_feature_data.append(features)
            labels = label.numpy()
            all_label_data.append(labels)
        all_feature_data = np.concatenate(all_feature_data, axis=0)
        self.scaler.fit(all_feature_data)
        all_label_data = np.concatenate(all_label_data, axis=0).reshape(-1, 1)
        self.label_scaler.fit(all_label_data)
        os.makedirs('models', exist_ok=True)
        with open(SCALER_FILE_NAME, 'wb') as f:
            pickle.dump((self.scaler, self.label_scaler), f)
        log.success('Scaler fitted and saved!')

    def build_model(self):
        log.log('Building model...')
        input_size = len(self.features)
        output_size = self.output_days
        model_class = MODEL_BLOCKS.get(self.model_type)
        if self.model_type == 'tcn':
            self.model = model_class(
                input_size=input_size,
                num_channels=[self.hidden_size] * 3,
                kernel_size=2,
                dropout=0.2,
                output_size=output_size
            )
        elif self.model_type == 'transformer':
            self.model = model_class(
                input_size=input_size,
                num_layers=3,
                nhead=13,
                hidden_size=self.hidden_size,
                output_size=output_size,
                dropout=0.1
            )
        elif self.model_type == 'cnn':
            self.model = model_class(
                input_size=input_size,
                num_filters=self.hidden_size,
                kernel_size=3,
                output_size=output_size,
                dropout=0.2
            )
        else:
            self.model = model_class(
                input_size=input_size,
                hidden_size=self.hidden_size,
                output_size=output_size,
                num_layers=2,
                dropout=0.3
            )
        self.model.to(device)
        self.criterion = CustomLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.grad_scaler = GradScaler(device.type)
        log.success('Model built and ready!')

    def save_model(self):
        log.log('Saving model...')
        os.makedirs('models', exist_ok=True)
        model_path = f"models/{self.model_name}.pth"
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_nickname': self._model_nickname,
            'epoch': self.epoch + 1
        }
        torch.save(checkpoint, model_path)
        log.success(f'Model saved to {model_path}!')

    def load_model(self):
        log.log('Loading model...')
        model_path = f"models/{self.model_name}.pth"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint.get('epoch', 0)
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self._model_nickname = checkpoint.get('model_nickname', 'UnnamedModel')
            self.model.to(device)
            log.success(f"Loaded model '{self.model_name}' from epoch {self.epoch}.")
        else:
            log.alert(f"No checkpoint found at '{model_path}'. Starting fresh.")

    def save_model_stats(self, all_stats: Dict[str, Any]):
        log.log('Saving model stats...')
        all_stats[self.model_name] = {
            'model_nickname': self._model_nickname,
            'model_type': self.model_type,
            'input_days': self.input_days,
            'output_days': self.output_days,
            'num_features': len(self.features),
            'hidden_size': self.hidden_size,
            'training_date_range': self.date_range,
            'epoch': self.epoch + 1,
            'best_train_loss': min(self.train_losses) if self.train_losses else None,
            'best_val_loss': min(self.val_losses) if self.val_losses else None
        }
        with open(MODEL_STATS_FILE_NAME, 'w') as json_file:
            json.dump(all_stats, json_file, indent=4)
        log.success('Model stats saved!')

    def load_model_stats(self) -> Dict[str, Any]:
        if os.path.exists(MODEL_STATS_FILE_NAME):
            with open(MODEL_STATS_FILE_NAME, 'r') as json_file:
                data = json.load(json_file)
            return data
        else:
            return {}

    def _generate_model_name(self) -> str:
        log.log('Generating model name...')
        model_name = self.model_type.upper()
        model_name += f'_HS{self.hidden_size}'
        model_name += f'_I{self.input_days}O{self.output_days}'
        model_name += f'_{datetime.now().strftime("%Y%m%d%H%M")}'
        log.success(f'Model name {model_name} generated!')
        return model_name

    def _generate_model_nickname(self, existing_nicknames: List[str]) -> str:
        log.log('Generating model nickname...')
        celestial_objects = [
            "Orion", "Andromeda", "Pegasus", "Cassiopeia", "Lyra",
            "Cygnus", "Draco", "Phoenix", "Hydra", "Scorpius", "Taurus", "Gemini",
            "Perseus", "Sagittarius", "Capricornus", "Aquarius", "Pisces",
            "Leo", "Virgo", "Cancer", "Libra", "Gemini",
            "Crux", "Corona", "Hercules", "Monoceros", "Canis Major",
            "Canis Minor", "Cetus", "Lepus", "Auriga", "Ophiuchus"
        ]
        model_type_list = list(self.model_types.keys())
        try:
            idx = model_type_list.index(self.model_type)
            celestial_object = celestial_objects[idx % len(celestial_objects)]
        except ValueError:
            celestial_object = "UnknownObject"
        count = sum(1 for name in existing_nicknames if celestial_object.lower() in name.lower())
        nickname = f"{celestial_object}{count + 1}"
        log.success(f'Model nickname {nickname} generated!')
        return nickname

    def prep_data(
        self,
        batch_size: int,
        num_workers: int = 8,
        pin_memory: bool = True,
        val_split: float = 0.1,
        test_split: float = 0.1,
        shuffle_test: bool = True
    ):
        log.log('Preparing data loaders...')
        self.dataset.set_scaler(self.scaler, self.label_scaler)
        total_size = len(self.dataset)
        test_size = int(total_size * test_split)
        val_size = int(total_size * val_split)
        train_size = total_size - test_size - val_size
        train_dataset = torch.utils.data.Subset(self.dataset, range(train_size))
        val_dataset = torch.utils.data.Subset(self.dataset, range(train_size, train_size + val_size))
        test_dataset = torch.utils.data.Subset(self.dataset, range(train_size + val_size, total_size))
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=pin_memory, drop_last=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=pin_memory, drop_last=False
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers,
            pin_memory=pin_memory, drop_last=False
        )
        log.success("Data loaders prepared!")

    def _run_through_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ):
        with autocast(device.type):
            output = self.model(inputs)
            loss = self.criterion(output, targets)
        return output, loss

    def train(self, num_epochs: int, batch_size: int, lr_scheduler: Literal['rltop', 'cawr']):
        log.log('Training started.')
        self.prep_data(batch_size)
        steps_per_epoch = len(self.train_loader)
        if lr_scheduler == 'rltop':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode='min',
                factor=0.0001,
                patience=3
            )
        elif lr_scheduler == 'cawr':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=1,
                eta_min=1e-6
            )
        all_stats = self.load_model_stats()
        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
                inputs, targets = self._prepare_batch(inputs, targets)
                self.optimizer.zero_grad()
                output, loss = self._run_through_loss(inputs, targets)
                self.grad_scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)
            avg_val_loss = self.validate(self.val_loader)
            self.val_losses.append(avg_val_loss)
            log.log(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f} - Unnormalized TL: {total_train_loss:.6f}")
            if lr_scheduler == 'rltop':
                self.scheduler.step(total_train_loss)
            else:
                self.scheduler.step()
            self.save_model()
            self.save_model_stats(all_stats)
            self.epoch += 1
        self.plot_losses()
        self.test_model()

    def _prepare_batch(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs_tensor = inputs.to(device).float()
        targets_tensor = targets.to(device).float()
        return inputs_tensor, targets_tensor

    def validate(self, val_loader: DataLoader) -> float:
        log.log('Validating model...')
        self.model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc='Validating'):
                inputs, targets = self._prepare_batch(inputs, targets)
                output, loss = self._run_through_loss(inputs, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        log.success('Validation complete!')
        return avg_val_loss

    def test_model(self):
        if not hasattr(self, 'test_loader'):
            log.alert("Test loader not found... Preparing loader...")
            self.prep_data(64, shuffle_test=True)
        log.log('Testing model...')
        self.model.eval()
        total_test_loss = 0.0
        all_targets = []
        all_predictions = []
        all_inputs = []
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="Testing"):
                inputs, targets = self._prepare_batch(inputs, targets)
                output, loss = self._run_through_loss(inputs, targets)
                total_test_loss += loss.item()
                all_targets.append(targets.cpu().float().numpy())
                all_predictions.append(output.cpu().float().numpy())
                all_inputs.append(inputs.cpu().float().numpy())
        avg_test_loss = total_test_loss / len(self.test_loader)
        log.log(f"Test Loss (MSE): {avg_test_loss:.6f}")
        self.evaluate_metrics(all_targets, all_predictions)
        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_inputs = np.concatenate(all_inputs, axis=0)
        self.plot_test_results(all_targets, all_predictions, all_inputs)
        log.success('Model testing complete!')

    def evaluate_metrics(self, targets: List[np.ndarray], predictions: List[np.ndarray]):
        targets = np.concatenate(targets, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        mae = np.mean(np.abs(targets - predictions))
        mse = np.mean((targets - predictions) ** 2)
        rmse = np.sqrt(mse)
        log.log(f"Test MAE: {mae:.6f}")
        log.log(f"Test RMSE: {rmse:.6f}")
        naive_predictions = np.zeros_like(targets)
        naive_predictions[:, :] = targets[:, 0][:, np.newaxis]
        naive_mse = np.mean((targets - naive_predictions) ** 2)
        naive_rmse = np.sqrt(naive_mse)
        log.log(f"Naive RMSE: {naive_rmse:.6f}")

    def plot_test_results(self, targets: np.ndarray, predictions: np.ndarray, inputs: np.ndarray):
        num_samples = min(5, len(targets))
        for i in range(num_samples):
            plt.figure(figsize=(10, 5))
            close_idx = self.features.index('Close')
            input_unscaled = self.scaler.inverse_transform(inputs[i])
            past_prices = input_unscaled[:, close_idx]
            target_unscaled = self.label_scaler.inverse_transform(targets[i].reshape(-1, 1)).flatten()
            prediction_unscaled = self.label_scaler.inverse_transform(predictions[i].reshape(-1, 1)).flatten()
            future_days = range(len(past_prices), len(past_prices) + self.output_days)
            plt.plot(range(len(past_prices)), past_prices, label='Past Prices', marker='o')
            plt.plot(future_days, target_unscaled, label='Actual Future Prices', marker='o')
            plt.plot(future_days, prediction_unscaled, label='Predicted Future Prices', marker='x')
            plt.xlabel('Day')
            plt.ylabel('Stock Price')
            plt.title(f'Test Sample {i+1}: Past and Future Stock Prices')
            plt.legend()
            plt.grid(True)
            plt.show()

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, label='Training Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

    def predict(self, ticker: str, date: str) -> Optional[Dict[str, Any]]:
        self.model.eval()
        print('Testing')
        log.log(f"Predicting for ticker {ticker} on date {date}...")
        date_dt = datetime.strptime(date, '%Y-%m-%d')
        date_range = [(date_dt - timedelta(days=self.input_days - 1)).strftime('%Y-%m-%d'), date]
        date_range_pd = pd.date_range(date_range[0], date_range[1])
        FetchStock.fetch_stock_data(ticker, date)
        FetchFred.fetch_fred_data(date)
        FetchSentiment.fetch_sentiment_data(ticker, date, back_up_days=self.input_days + 15)
        stock_data = {}
        fred_data = {}
        sentiment_data = {}
        for days in date_range_pd:
            days = days.strftime('%Y-%m-%d')
            stock_data[days] = FetchStock.fetch_stock_data(ticker, days)
            fred_data[days] = FetchFred.fetch_fred_data(days)
            sentiment_data[days] = FetchSentiment.fetch_sentiment_data(ticker, days)
        input_prices = {}
        for days in stock_data:
            input_prices[days] = stock_data[days]['Close']
        input_data = StockDataset.combine_data_blank(stock_data, fred_data, sentiment_data)
        input_data_fixed = StockDataset.prepare_input_blank(input_data)
        input_features = self.scaler.transform(input_data_fixed)
        state = torch.tensor(input_features, dtype=torch.float32)
        state = torch.nan_to_num(state, nan=0)
        state = state.unsqueeze(0)
        outputs = self.model(state)
        outputs = outputs.detach().cpu().numpy()
        outputs_unscaled = self.label_scaler.inverse_transform(outputs)
        predictions = outputs_unscaled.flatten()
        predicted_dates = [(date_dt + timedelta(days=i + 1)).strftime('%Y-%m-%d') for i in range(self.output_days)]
        input_dates = [d.strftime('%Y-%m-%d') for d in date_range_pd]
        prediction_dict = {
            'predicted_prices': {predicted_dates[i]: predictions[i] for i in range(len(predictions))},
            'input_prices': input_prices,
            'input_dates': input_dates
        }
        return prediction_dict

if __name__ == '__main__':
    model_type = 'lstm'
    stock_list = ticker_symbols
    input_days = 15
    output_days = 3
    lr = 0.001
    predictor = StockPredictor(
        model_type=model_type,
        stock_list=stock_list,
        input_days=input_days,
        output_days=output_days,
        model_name='LSTM_HS128_I15O3_202410220855'
    )
    predictor.train(100, 64, 'rltop')