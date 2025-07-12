import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import joblib
import sys
import time
import datetime
from tqdm import tqdm
from colorama import Fore, Style
import torch.nn.functional as F

try:
    from .. import config
    from .architecture import UniversalTransformerModel
    from ..model_manager import ModelManager
except (ImportError, ValueError):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src import config
    from src.modeling.architecture import UniversalTransformerModel
    from src.model_manager import ModelManager

class AsymmetricMSELoss(nn.Module):
    def __init__(self, alpha=0.6):
        super().__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        pos_diff = F.relu(diff)
        neg_diff = F.relu(-diff)
        
        loss = torch.mean(torch.log(torch.cosh(pos_diff)) * self.alpha + 
                         torch.log(torch.cosh(neg_diff)) * (1 - self.alpha))
        return loss

class UniversalModelEngine:
    def __init__(self, data, model_config=None):
        self.data = data
        self.device = torch.device(config.DEVICE)
        self.feature_cols = config.TECHNICAL_FEATURES + config.TIME_FEATURES + config.MACRO_FEATURES
        self.tickers = self.data['ticker'].unique()
        self.ticker_map = {ticker: i for i, ticker in enumerate(self.tickers)}
        self.model_manager = ModelManager()
        self.training_start_time = None
        
        current_model_config = model_config if model_config is not None else config.MODEL_CONFIG

        self.model = UniversalTransformerModel(
            input_dim=len(self.feature_cols),
            ticker_count=len(self.tickers),
            d_model=current_model_config['d_model'],
            n_heads=current_model_config['n_heads'],
            n_layers=current_model_config['n_layers'],
            dropout=current_model_config['dropout'],
            ticker_embedding_dim=current_model_config['ticker_embedding_dim'],
            sequence_length=current_model_config['sequence_length'],
            prediction_horizon=current_model_config['prediction_horizon']
        ).to(self.device)
        self.temp_model_path = os.path.join(config.MODELS_PATH, 'temp_model.pth')
        self.temp_scaler_path = os.path.join(config.MODELS_PATH, 'temp_scaler.pkl')
        self.checkpoint_path = config.CHECKPOINT_PATH

    def _create_sequences(self, data, ticker_ids):
        seq_len = config.MODEL_CONFIG['sequence_length']
        pred_hor = config.MODEL_CONFIG['prediction_horizon']
        n_sequences = len(data) - seq_len - pred_hor + 1
        
        if n_sequences <= 0:
            return np.array([]), np.array([]), np.array([])
        
        sequences = np.empty((n_sequences, seq_len, data.shape[1]), dtype=np.float32)
        targets = np.empty((n_sequences, pred_hor, data.shape[1]), dtype=np.float32)
        ids = np.empty(n_sequences, dtype=np.int64)
        
        for i in range(n_sequences):
            sequences[i] = data[i:i + seq_len]
            targets[i] = data[i + seq_len:i + seq_len + pred_hor]
            ids[i] = ticker_ids[i]
        
        return sequences, targets, ids

    def prepare_data(self):
        print("Preparing data...")
        
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        
        if os.path.exists(self.temp_scaler_path):
            self.scaler = joblib.load(self.temp_scaler_path)
            data_scaled = self.scaler.transform(self.data[self.feature_cols])
        else:
            data_scaled = self.scaler.fit_transform(self.data[self.feature_cols])
        
        self.data.loc[:, self.feature_cols] = data_scaled
        
        total_sequences = 0
        for ticker in self.tickers:
            ticker_data = self.data[self.data['ticker'] == ticker]
            seq_len = config.MODEL_CONFIG['sequence_length']
            pred_hor = config.MODEL_CONFIG['prediction_horizon']
            n_seq = len(ticker_data) - seq_len - pred_hor + 1
            if n_seq > 0:
                total_sequences += n_seq
        
        if total_sequences == 0:
            raise ValueError("No valid sequences can be created from the data")
        
        X = np.empty((total_sequences, seq_len, len(self.feature_cols)), dtype=np.float32)
        y = np.empty((total_sequences, pred_hor, len(self.feature_cols)), dtype=np.float32)
        ids = np.empty(total_sequences, dtype=np.int64)
        
        idx = 0
        for ticker in self.tickers:
            ticker_data = self.data[self.data['ticker'] == ticker]
            ticker_id_val = self.ticker_map[ticker]
            ticker_features = ticker_data[self.feature_cols].values
            ticker_ids_array = np.full(len(ticker_features), ticker_id_val, dtype=np.int64)
            
            sequences, targets, seq_ids = self._create_sequences(ticker_features, ticker_ids_array)
            if len(sequences) > 0:
                end_idx = idx + len(sequences)
                X[idx:end_idx] = sequences
                y[idx:end_idx] = targets
                ids[idx:end_idx] = seq_ids
                idx = end_idx
        
        X = X[:idx]
        y = y[:idx]
        ids = ids[:idx]
        
        indices = np.random.permutation(len(X))
        X, y, ids = X[indices], y[indices], ids[indices]
        
        val_size = int(len(X) * config.TRAIN_CONFIG['val_split_ratio'])
        train_size = len(X) - val_size
        
        train_data = TensorDataset(
            torch.from_numpy(X[:train_size]).to(self.device, dtype=torch.float32), 
            torch.from_numpy(y[:train_size]).to(self.device, dtype=torch.float32), 
            torch.from_numpy(ids[:train_size]).to(self.device, dtype=torch.long)
        )
        val_data = TensorDataset(
            torch.from_numpy(X[train_size:]).to(self.device, dtype=torch.float32), 
            torch.from_numpy(y[train_size:]).to(self.device, dtype=torch.float32), 
            torch.from_numpy(ids[train_size:]).to(self.device, dtype=torch.long)
        )
        
        self.train_loader = DataLoader(
            train_data, 
            batch_size=config.TRAIN_CONFIG['batch_size'], 
            shuffle=True, 
            pin_memory=False,
            num_workers=0
        )
        self.val_loader = DataLoader(
            val_data, 
            batch_size=config.TRAIN_CONFIG['batch_size'], 
            shuffle=False, 
            pin_memory=False,
            num_workers=0
        )
        
        print(f"Data prepared: {train_size} train, {val_size} val samples")

    def _save_checkpoint(self, epoch, optimizer, scheduler, val_loss, is_best=False):
        if epoch % config.TRAIN_CONFIG['checkpoint_every'] == 0 or is_best:
            try:
                state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'val_loss': val_loss,
                    'scaler_state': self.scaler,
                    'feature_cols': self.feature_cols,
                    'tickers': list(self.tickers),
                    'ticker_map': self.ticker_map,
                    'model_config': config.MODEL_CONFIG,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'scheduler_step': scheduler.last_epoch if scheduler else 0
                }
                
                os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
                torch.save(state, self.checkpoint_path)
                
                if is_best:
                    best_path = self.temp_model_path.replace('.pth', '_best.pth')
                    torch.save(state, best_path)
                    
            except Exception as e:
                print(f"Warning: Failed to save checkpoint: {e}")

    def _load_checkpoint(self, optimizer, scheduler):
        if os.path.exists(self.checkpoint_path):
            print(f"Loading checkpoint from: {self.checkpoint_path}")
            try:
                from sklearn.preprocessing import StandardScaler
                torch.serialization.add_safe_globals([StandardScaler])
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    scheduler.last_epoch = checkpoint.get('scheduler_step', 0)
                
                start_epoch = checkpoint['epoch'] + 1
                best_val_loss = checkpoint.get('val_loss', float('inf'))
                
                if 'scaler_state' in checkpoint:
                    self.scaler = checkpoint['scaler_state']
                
                print(f"Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.6f}")
                return start_epoch, best_val_loss
                
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                return 0, float('inf')
        else:
            print("No checkpoint found, starting fresh training")
            return 0, float('inf')

    def train(self):
        if not os.path.exists(config.MODELS_PATH):
            os.makedirs(config.MODELS_PATH)
        if not os.path.exists(config.SAVED_MODELS_PATH):
            os.makedirs(config.SAVED_MODELS_PATH)
        
        self.training_start_time = time.time()
        print("Starting training...")
        self.prepare_data()
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.TRAIN_CONFIG['learning_rate'],
            weight_decay=config.TRAIN_CONFIG['weight_decay']
        )
        
        start_epoch, best_val_loss = self._load_checkpoint(optimizer, None)
        
        total_steps = len(self.train_loader) * (config.TRAIN_CONFIG['epochs'] - start_epoch)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.TRAIN_CONFIG['max_lr'],
            total_steps=total_steps,
            pct_start=config.TRAIN_CONFIG['warmup_epochs'] / config.TRAIN_CONFIG['epochs'],
            anneal_strategy='cos',
            cycle_momentum=False,
            div_factor=10.0,
            final_div_factor=100.0
        )
        
        if start_epoch > 0:
            for _ in range(start_epoch * len(self.train_loader)):
                if scheduler.last_epoch < total_steps - 1:
                    scheduler.step()
        
        print(f"Training on {config.DEVICE} device")
        print(f"Starting from epoch {start_epoch}")
        print(f"Total steps: {total_steps}, Steps per epoch: {len(self.train_loader)}")
        
        patience_counter = 0
        loss_history = []
        grad_norm_sum = 0
        epoch = start_epoch
        
        for epoch in tqdm(range(start_epoch, config.TRAIN_CONFIG['epochs']), desc="Training Progress"):
            self.model.train()
            total_train_loss = 0
            batch_count = 0
            
            for seq, target, ids in tqdm(self.train_loader, desc=f"Epoch {epoch+1}", disable=True):
                optimizer.zero_grad()
                output = self.model(seq, ids)
                loss = criterion(output, target)
                loss.backward()
                
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.TRAIN_CONFIG['gradient_clip_val'])
                grad_norm_sum += total_norm.item()
                batch_count += 1
                
                optimizer.step()
                if scheduler.last_epoch < total_steps - 1:
                    scheduler.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(self.train_loader)
            avg_grad_norm = grad_norm_sum / batch_count if batch_count > 0 else 0
            grad_norm_sum = 0
            
            if len(loss_history) >= 20:
                loss_history.pop(0)
            loss_history.append(avg_train_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            
            if avg_grad_norm > 5.0 or avg_train_loss > 100.0:
                print(f"Training instability detected! Grad norm: {avg_grad_norm:.2f}, Loss: {avg_train_loss:.2f}")
                print("Reducing learning rate and continuing...")
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                continue
            
            if len(loss_history) > 5:
                recent_losses = loss_history[-5:]
                if all(loss > recent_losses[0] * 2 for loss in recent_losses[1:]):
                    print("Loss explosion detected! Reducing learning rate...")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.1
                    continue
            
            self.model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for seq, target, ids in self.val_loader:
                    output = self.model(seq, ids)
                    loss = criterion(output, target)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(self.val_loader)
            
            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.temp_model_path)
                joblib.dump(self.scaler, self.temp_scaler_path)
                print(f"New best model saved! Val Loss: {avg_val_loss:.6f}")
            else:
                patience_counter += 1
            
            self._save_checkpoint(epoch, optimizer, scheduler, avg_val_loss, is_best)
            
            print(f"Epoch {epoch+1}/{config.TRAIN_CONFIG['epochs']} | "
                  f"Train: {avg_train_loss:.4f} | "
                  f"Val: {avg_val_loss:.4f} | "
                  f"Best: {best_val_loss:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"GradNorm: {avg_grad_norm:.3f} | "
                  f"Patience: {patience_counter}")
            
            if patience_counter >= config.TRAIN_CONFIG['early_stopping_patience']:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
        
        training_time = time.time() - self.training_start_time
        print(f"Training complete! Best Val Loss: {best_val_loss:.6f}")
        print(f"Total training time: {training_time:.2f} seconds")
        
        model_folder = self._save_completed_model(best_val_loss, epoch, training_time)
        return model_folder

    def _save_completed_model(self, best_val_loss, epochs_trained, training_time):
        try:
            data = pd.read_parquet(config.DATA_PATH)
            current_tickers = data['ticker'].unique().tolist()
        except:
            current_tickers = list(self.tickers)
        
        metadata = {
            'created_at': datetime.datetime.now().isoformat(),
            'model_type': 'UniversalTransformerModel',
            'model_architecture': 'UniversalTransformerModel',
            'best_val_loss': float(best_val_loss),
            'final_val_loss': float(best_val_loss),
            'epochs_trained': int(epochs_trained + 1),
            'training_time': f"{training_time:.2f}s",
            'device': str(config.DEVICE),
            'model_config': dict(config.MODEL_CONFIG),
            'train_config': dict(config.TRAIN_CONFIG),
            'feature_count': len(self.feature_cols),
            'ticker_count': len(current_tickers),
            'tickers': current_tickers,
            'feature_columns': list(self.feature_cols),
            'target_feature': config.TARGET_FEATURE,
            'accuracy_metrics': {},
            'data_path': config.DATA_PATH,
            'version': '1.0'
        }
        
        model_folder = self.model_manager.save_completed_model(self.temp_model_path, self.temp_scaler_path, metadata)
        
        if os.path.exists(self.temp_model_path):
            os.remove(self.temp_model_path)
        if os.path.exists(self.temp_scaler_path):
            os.remove(self.temp_scaler_path)
            
        return model_folder

    def predict(self, ticker, horizon, model_path=None, scaler_path=None):
        if model_path and scaler_path:
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Model or scaler not found: {model_path}, {scaler_path}")
            
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.scaler = joblib.load(scaler_path)
        elif hasattr(self, 'scaler') and self.scaler is not None:
            pass
        elif hasattr(self, 'temp_model_path') and hasattr(self, 'temp_scaler_path') and os.path.exists(self.temp_model_path) and os.path.exists(self.temp_scaler_path):
            self.model.load_state_dict(torch.load(self.temp_model_path, map_location=self.device))
            self.scaler = joblib.load(self.temp_scaler_path)
        else:
            raise FileNotFoundError("No model path provided and no trained model found. Please train the model first.")

        self.model.eval()

        ticker_data = self.data[self.data['ticker'] == ticker]
        if len(ticker_data) < self.model.sequence_length:
            raise ValueError(f"Insufficient data for ticker {ticker}")
        
        last_sequence_raw = ticker_data[self.feature_cols].values[-self.model.sequence_length:]
        last_sequence_scaled = self.scaler.transform(last_sequence_raw)

        input_seq = torch.from_numpy(last_sequence_scaled).unsqueeze(0).to(self.device, dtype=torch.float32)
        ticker_id = torch.tensor([self.ticker_map[ticker]], device=self.device, dtype=torch.long)

        predictions = torch.empty((horizon, len(self.feature_cols)), device=self.device, dtype=torch.float32)
        
        with torch.no_grad():
            for i in range(horizon):
                prediction_scaled = self.model(input_seq, ticker_id)
                next_pred_scaled = prediction_scaled[:, -1, :]
                predictions[i] = next_pred_scaled.squeeze(0)
                input_seq = torch.cat([input_seq[:, 1:, :], next_pred_scaled.unsqueeze(1)], dim=1)

        predictions_cpu = predictions.cpu().numpy()
        predictions_unscaled = self.scaler.inverse_transform(predictions_cpu)

        return pd.DataFrame(predictions_unscaled, columns=self.feature_cols)