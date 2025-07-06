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
        loss = torch.mean(F.relu(diff)**2 * self.alpha + F.relu(-diff)**2 * (1 - self.alpha))
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
        self.model_path = os.path.join(config.MODELS_PATH, 'universal_model.pth')
        self.scaler_path = os.path.join(config.MODELS_PATH, 'universal_scaler.pkl')
        self.checkpoint_path = config.CHECKPOINT_PATH

    def _create_sequences(self, data, ticker_ids):
        sequences, targets, ids = [], [], []
        seq_len = config.MODEL_CONFIG['sequence_length']
        pred_hor = config.MODEL_CONFIG['prediction_horizon']
        for i in range(len(data) - seq_len - pred_hor + 1):
            sequences.append(data[i:i + seq_len])
            targets.append(data[i + seq_len:i + seq_len + pred_hor])
            ids.append(ticker_ids[i])
        return np.array(sequences), np.array(targets), np.array(ids)

    def prepare_data(self):
        print("Preparing data...")
        
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            data_scaled = self.scaler.transform(self.data[self.feature_cols])
        else:
            data_scaled = self.scaler.fit_transform(self.data[self.feature_cols])
            joblib.dump(self.scaler, self.scaler_path)
        
        self.data[self.feature_cols] = data_scaled
        
        all_sequences, all_targets, all_ids = [], [], []
        
        for ticker in self.tickers:
            ticker_data = self.data[self.data['ticker'] == ticker]
            ticker_id_val = self.ticker_map[ticker]
            ticker_features = ticker_data[self.feature_cols].values
            ticker_ids_array = np.full(len(ticker_features), ticker_id_val)
            
            sequences, targets, ids = self._create_sequences(ticker_features, ticker_ids_array)
            if len(sequences) > 0:
                all_sequences.append(sequences)
                all_targets.append(targets)
                all_ids.append(ids)
        
        X = np.concatenate(all_sequences)
        y = np.concatenate(all_targets)
        ids = np.concatenate(all_ids)
        
        indices = np.random.permutation(len(X))
        X, y, ids = X[indices], y[indices], ids[indices]
        
        total_samples = len(X)
        val_size = int(total_samples * config.TRAIN_CONFIG['val_split_ratio'])
        train_size = total_samples - val_size
        
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        ids_train, ids_val = ids[:train_size], ids[train_size:]
        
        train_data = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train), 
            torch.LongTensor(ids_train)
        )
        val_data = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.FloatTensor(y_val), 
            torch.LongTensor(ids_val)
        )
        
        self.train_loader = DataLoader(train_data, batch_size=config.TRAIN_CONFIG['batch_size'], shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=config.TRAIN_CONFIG['batch_size'], shuffle=False)
        
        print(f"Data prepared: {len(X_train)} train, {len(X_val)} val samples")

    def _save_checkpoint(self, epoch, optimizer, scheduler, val_loss, is_best=False):
        if epoch % config.TRAIN_CONFIG['checkpoint_every'] == 0 or is_best:
            state = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'scaler_state': self.scaler
            }
            if scheduler is not None:
                state['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(state, self.checkpoint_path)
            if is_best:
                torch.save(state, self.model_path.replace('.pth', '_best.pth'))

    def _load_checkpoint(self, optimizer, scheduler):
        if os.path.exists(self.checkpoint_path):
            print(f"Loading checkpoint from: {self.checkpoint_path}")
            try:
                from sklearn.preprocessing import StandardScaler
                torch.serialization.add_safe_globals([StandardScaler])
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
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
        
        self.training_start_time = time.time()
        print("Starting training...")
        self.prepare_data()
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.TRAIN_CONFIG['learning_rate'])
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.TRAIN_CONFIG['max_lr'],  # Use configured max_lr
            steps_per_epoch=len(self.train_loader),
            epochs=config.TRAIN_CONFIG['epochs'],
            pct_start=config.TRAIN_CONFIG['warmup_epochs'] / config.TRAIN_CONFIG['epochs'],  # Dynamic warmup
            anneal_strategy='cos',
            cycle_momentum=False
        )
        
        start_epoch, best_val_loss = self._load_checkpoint(optimizer, scheduler)
        
        print(f"Training on {config.DEVICE} device")
        print(f"Starting from epoch {start_epoch}")
        
        patience_counter = 0
        loss_history = []
        grad_norm_history = []
        epoch = start_epoch
        
        for epoch in tqdm(range(start_epoch, config.TRAIN_CONFIG['epochs']), desc="Training Progress"):
            self.model.train()
            total_train_loss = 0
            epoch_grad_norms = []
            
            for seq, target, ids in tqdm(self.train_loader, desc=f"Epoch {epoch+1}"):
                seq = seq.to(self.device)
                target = target.to(self.device)
                ids = ids.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(seq, ids)
                loss = criterion(output, target)
                loss.backward()
                
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.TRAIN_CONFIG['gradient_clip_val'])
                epoch_grad_norms.append(total_norm.item())
                
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(self.train_loader)
            avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)
            
            loss_history.append(avg_train_loss)
            grad_norm_history.append(avg_grad_norm)
            
            if len(loss_history) > 20:
                loss_history.pop(0)
                grad_norm_history.pop(0)
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # Check for training instability
            if avg_grad_norm > 5.0 or avg_train_loss > 100.0:
                print(f"Training instability detected! Grad norm: {avg_grad_norm:.2f}, Loss: {avg_train_loss:.2f}")
                print("Reducing learning rate and continuing...")
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                continue
            
            # Check for loss explosion
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
                    seq = seq.to(self.device)
                    target = target.to(self.device)
                    ids = ids.to(self.device)
                    
                    output = self.model(seq, ids)
                    loss = criterion(output, target)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(self.val_loader)
            
            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_path)
                joblib.dump(self.scaler, self.scaler_path)
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
        metadata = {
            'created_at': datetime.datetime.now().isoformat(),
            'model_type': 'UniversalTransformerModel',
            'best_val_loss': float(best_val_loss),
            'epochs_trained': epochs_trained + 1,
            'training_time': f"{training_time:.2f}s",
            'device': config.DEVICE,
            'model_config': config.MODEL_CONFIG,
            'train_config': config.TRAIN_CONFIG,
            'feature_count': len(self.feature_cols),
            'ticker_count': len(self.tickers),
            'tickers': list(self.tickers),
            'feature_columns': self.feature_cols,
            'accuracy_metrics': {}
        }
        
        model_folder = self.model_manager.save_completed_model(self.model_path, self.scaler_path, metadata)
        return model_folder

    def predict(self, ticker, horizon):
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError("Trained model or scaler not found. Please train the model first.")

        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        self.scaler = joblib.load(self.scaler_path)

        ticker_data = self.data[self.data['ticker'] == ticker].copy()
        last_sequence_raw = ticker_data[self.feature_cols].values[-self.model.sequence_length:]
        last_sequence_scaled = self.scaler.transform(last_sequence_raw)

        input_seq = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).to(self.device)
        ticker_id = torch.LongTensor([self.ticker_map[ticker]]).to(self.device)

        predictions = []
        with torch.no_grad():
            for _ in range(horizon):
                prediction_scaled = self.model(input_seq, ticker_id)
                
                next_pred_scaled = prediction_scaled[:, -1, :].unsqueeze(1)
                predictions.append(next_pred_scaled)
                
                input_seq = torch.cat([input_seq[:, 1:, :], next_pred_scaled], dim=1)

        predictions_scaled = torch.cat(predictions, dim=1).cpu().numpy()
        predictions_scaled = predictions_scaled.reshape(horizon, -1)
        predictions_unscaled = self.scaler.inverse_transform(predictions_scaled)

        return pd.DataFrame(predictions_unscaled, columns=self.feature_cols)