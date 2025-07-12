import os
import json
import shutil
import datetime
import torch
import joblib
from pathlib import Path
from colorama import Fore, Style
import pandas as pd
import numpy as np

try:
    from . import config
    from .backup_manager import BackupManager
except (ImportError, ValueError):
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src import config
    from src.backup_manager import BackupManager

class ModelManager:
    def __init__(self):
        self.saved_models_path = Path(config.SAVED_MODELS_PATH)
        self.saved_models_path.mkdir(exist_ok=True)
        self.backup_manager = BackupManager()
    
    def save_completed_model(self, model_path, scaler_path, metadata):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_folder = self.saved_models_path / f"model_{timestamp}"
        model_folder.mkdir(exist_ok=True)
        
        try:
            if os.path.exists(model_path):
                shutil.copy2(model_path, model_folder / "model.pth")
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            if os.path.exists(scaler_path):
                shutil.copy2(scaler_path, model_folder / "scaler.pkl")
            else:
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            
            metadata = self._ensure_complete_metadata(metadata)
            
            metadata_path = model_folder / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=self._json_serializer)
            
            backup_path = self.backup_manager.auto_backup_strategy(str(model_folder))
            print(f"{Fore.GREEN}Model saved to: {model_folder}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Backup created: {backup_path}{Style.RESET_ALL}")
            return str(model_folder)
            
        except Exception as e:
            if model_folder.exists():
                shutil.rmtree(model_folder)
            raise Exception(f"Failed to save model: {str(e)}")
    
    def restore_model_from_backup(self, backup_path, restore_name=None):
        try:
            if restore_name is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                restore_name = f"restored_model_{timestamp}"
            
            restore_location = self.saved_models_path / restore_name
            restored_path = self.backup_manager.restore_from_backup(backup_path, restore_location)
            
            if self._validate_model_files(Path(restored_path)):
                print(f"{Fore.GREEN}Model successfully restored: {restored_path}{Style.RESET_ALL}")
                return restored_path
            else:
                print(f"{Fore.RED}Restored model failed validation{Style.RESET_ALL}")
                return None
                
        except Exception as e:
            print(f"{Fore.RED}Failed to restore model: {str(e)}{Style.RESET_ALL}")
            return None
    
    def create_model_backup(self, model_folder, backup_type="smart"):
        try:
            if backup_type == "full":
                return self.backup_manager.create_full_backup(model_folder)
            elif backup_type == "incremental":
                return self.backup_manager.create_incremental_backup(model_folder)
            elif backup_type == "compressed":
                return self.backup_manager.create_compressed_backup(model_folder)
            else:
                return self.backup_manager.auto_backup_strategy(model_folder, backup_type)
        except Exception as e:
            print(f"{Fore.RED}Failed to create backup: {str(e)}{Style.RESET_ALL}")
            return None
    
    def verify_model_integrity(self, model_folder):
        model_path = Path(model_folder)
        
        if not self._validate_model_files(model_path):
            return False
        
        try:
            model_file = model_path / "model.pth"
            scaler_file = model_path / "scaler.pkl"
            metadata_file = model_path / "metadata.json"
            
            checkpoint = torch.load(model_file, map_location='cpu', weights_only=True)
            if not isinstance(checkpoint, dict):
                print(f"{Fore.RED}Invalid model file format{Style.RESET_ALL}")
                return False
            
            scaler = joblib.load(scaler_file)
            if not hasattr(scaler, 'transform'):
                print(f"{Fore.RED}Invalid scaler file{Style.RESET_ALL}")
                return False
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            if not isinstance(metadata, dict):
                print(f"{Fore.RED}Invalid metadata file{Style.RESET_ALL}")
                return False
            
            print(f"{Fore.GREEN}Model integrity verified: {model_folder}{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Model integrity check failed: {str(e)}{Style.RESET_ALL}")
            return False
    
    def cleanup_old_models(self, keep_count=5):
        models = self.list_saved_models()
        if len(models) <= keep_count:
            print(f"{Fore.CYAN}No cleanup needed. Current models: {len(models)}/{keep_count}{Style.RESET_ALL}")
            return
        
        to_delete = models[keep_count:]
        for model in to_delete:
            try:
                model_path = Path(model['path'])
                if model_path.exists():
                    shutil.rmtree(model_path)
                    print(f"{Fore.YELLOW}Deleted old model: {model['name']}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Failed to delete model {model['name']}: {str(e)}{Style.RESET_ALL}")
        
        print(f"{Fore.GREEN}Cleanup complete. Deleted {len(to_delete)} old models{Style.RESET_ALL}")
    
    def list_backups(self):
        return self.backup_manager.list_backups()
    
    def verify_backup_integrity(self, backup_path):
        return self.backup_manager.verify_backup_integrity(backup_path)
    
    def cleanup_old_backups(self, keep_count=10, keep_days=30):
        self.backup_manager.cleanup_old_backups(keep_count, keep_days)
    
    def _json_serializer(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def _ensure_complete_metadata(self, metadata):
        required_fields = {
            'created_at': datetime.datetime.now().isoformat(),
            'model_type': 'UniversalTransformerModel',
            'best_val_loss': 0.0,
            'final_val_loss': 0.0,
            'epochs_trained': 0,
            'training_time': '0s',
            'device': config.DEVICE,
            'model_config': config.MODEL_CONFIG.copy(),
            'train_config': config.TRAIN_CONFIG.copy(),
            'feature_count': 0,
            'ticker_count': 0,
            'tickers': [],
            'feature_columns': [],
            'target_feature': config.TARGET_FEATURE,
            'accuracy_metrics': {},
            'model_architecture': 'UniversalTransformerModel',
            'data_path': config.DATA_PATH,
            'version': '1.0'
        }
        
        for key, default_value in required_fields.items():
            if key not in metadata or metadata[key] is None:
                metadata[key] = default_value
        
        if not metadata.get('tickers'):
            try:
                data = pd.read_parquet(config.DATA_PATH)
                metadata['tickers'] = data['ticker'].unique().tolist()
                metadata['ticker_count'] = len(metadata['tickers'])
            except:
                metadata['tickers'] = config.TICKERS
                metadata['ticker_count'] = len(config.TICKERS)
        
        if not metadata.get('feature_columns'):
            metadata['feature_columns'] = config.TECHNICAL_FEATURES + config.TIME_FEATURES + config.MACRO_FEATURES
            metadata['feature_count'] = len(metadata['feature_columns'])
        
        return metadata
    
    def copy_accuracy_analysis(self, model_folder, analysis_files):
        model_path = Path(model_folder)
        analysis_folder = model_path / "analysis"
        analysis_folder.mkdir(parents=True, exist_ok=True)
        
        for file_path in analysis_files:
            if os.path.exists(file_path):
                shutil.copy2(file_path, analysis_folder / os.path.basename(file_path))
    
    def list_saved_models(self):
        models = []
        for model_dir in self.saved_models_path.iterdir():
            if model_dir.is_dir() and model_dir.name.startswith("model_"):
                
                if not self.validate_and_migrate_model(model_dir):
                    continue
                
                if not self._validate_model_files(model_dir):
                    print(f"{Fore.YELLOW}Warning: Model {model_dir.name} has missing files{Style.RESET_ALL}")
                    continue
                
                metadata_path = model_dir / "metadata.json"
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    models.append({
                        'path': str(model_dir),
                        'name': model_dir.name,
                        'metadata': metadata
                    })
                except Exception as e:
                    print(f"{Fore.RED}Error loading metadata for {model_dir.name}: {str(e)}{Style.RESET_ALL}")
                    continue
        
        return sorted(models, key=lambda x: x['name'], reverse=True)
    
    def _validate_model_files(self, model_dir):
        required_files = ['model.pth', 'scaler.pkl', 'metadata.json']
        for file_name in required_files:
            if not (model_dir / file_name).exists():
                return False
        return True
    
    def load_model_for_prediction(self, model_folder):
        model_path = Path(model_folder)
        model_file = model_path / "model.pth"
        scaler_file = model_path / "scaler.pkl"
        metadata_file = model_path / "metadata.json"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        if not scaler_file.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_file}")
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load metadata: {str(e)}")
        
        return str(model_file), str(scaler_file), metadata
    
    def update_model_accuracy(self, model_folder, accuracy_metrics):
        metadata_path = Path(model_folder) / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                metadata['accuracy_metrics'] = accuracy_metrics
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=self._json_serializer)
                print(f"{Fore.GREEN}Updated model accuracy metrics{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Failed to update accuracy metrics: {str(e)}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}Metadata file not found for model update{Style.RESET_ALL}")
    
    def display_model_summary(self):
        models = self.list_saved_models()
        if not models:
            print(f"{Fore.YELLOW}No saved models found{Style.RESET_ALL}")
            return
        
        print(f"{Fore.CYAN}Available Saved Models:{Style.RESET_ALL}")
        print("-" * 80)
        
        for i, model in enumerate(models, 1):
            metadata = model['metadata']
            print(f"{Fore.WHITE}{i}. {model['name']}{Style.RESET_ALL}")
            print(f"   Created: {metadata.get('created_at', 'Unknown')}")
            
            final_val_loss = metadata.get('final_val_loss', metadata.get('best_val_loss', 'N/A'))
            best_val_loss = metadata.get('best_val_loss', 'N/A')
            
            if isinstance(final_val_loss, (int, float)):
                print(f"   Final Val Loss: {final_val_loss:.6f}")
            else:
                print(f"   Final Val Loss: {final_val_loss}")
                
            if isinstance(best_val_loss, (int, float)):
                print(f"   Best Val Loss: {best_val_loss:.6f}")
            else:
                print(f"   Best Val Loss: {best_val_loss}")
                
            print(f"   Epochs Trained: {metadata.get('epochs_trained', 'N/A')}")
            print(f"   Training Time: {metadata.get('training_time', 'N/A')}")
            
            accuracy = metadata.get('accuracy_metrics', {})
            if accuracy and any(isinstance(v, (int, float)) for v in accuracy.values()):
                mae = accuracy.get('mae', 0)
                rmse = accuracy.get('rmse', 0)
                r2 = accuracy.get('r2', 0)
                direction_accuracy = accuracy.get('direction_accuracy', 0)
                
                print(f"   MAE: {mae:.4f}")
                print(f"   RMSE: {rmse:.4f}")
                print(f"   R²: {r2:.4f}")
                print(f"   Direction Accuracy: {direction_accuracy:.2%}")
            else:
                print(f"   MAE: 0.0000")
                print(f"   RMSE: 0.0000")
                print(f"   R²: 0.0000")
                print(f"   Direction Accuracy: 0.00%")
            print()
    
    def validate_and_migrate_model(self, model_folder):
        model_path = Path(model_folder)
        metadata_file = model_path / "metadata.json"
        
        if not metadata_file.exists():
            print(f"{Fore.YELLOW}No metadata found for {model_path.name}, creating default metadata{Style.RESET_ALL}")
            self._create_default_metadata(model_path)
            return True
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            updated = False
            
            if 'version' not in metadata:
                metadata['version'] = '1.0'
                updated = True
            
            if 'target_feature' not in metadata:
                metadata['target_feature'] = config.TARGET_FEATURE
                updated = True
            
            if 'model_architecture' not in metadata:
                metadata['model_architecture'] = metadata.get('model_type', 'UniversalTransformerModel')
                updated = True
                
            if 'data_path' not in metadata:
                metadata['data_path'] = config.DATA_PATH
                updated = True
            
            if not metadata.get('feature_columns'):
                metadata['feature_columns'] = config.TECHNICAL_FEATURES + config.TIME_FEATURES + config.MACRO_FEATURES
                metadata['feature_count'] = len(metadata['feature_columns'])
                updated = True
            
            if not metadata.get('tickers'):
                try:
                    data = pd.read_parquet(config.DATA_PATH)
                    metadata['tickers'] = data['ticker'].unique().tolist()
                    metadata['ticker_count'] = len(metadata['tickers'])
                    updated = True
                except:
                    metadata['tickers'] = config.TICKERS
                    metadata['ticker_count'] = len(config.TICKERS)
                    updated = True
            
            if updated:
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=self._json_serializer)
                print(f"{Fore.GREEN}Updated metadata for {model_path.name}{Style.RESET_ALL}")
            
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Failed to validate/migrate model {model_path.name}: {e}{Style.RESET_ALL}")
            return False
    
    def _create_default_metadata(self, model_path):
        try:
            data = pd.read_parquet(config.DATA_PATH)
            tickers = data['ticker'].unique().tolist()
        except:
            tickers = config.TICKERS
        
        feature_columns = config.TECHNICAL_FEATURES + config.TIME_FEATURES + config.MACRO_FEATURES
        
        metadata = {
            'created_at': datetime.datetime.now().isoformat(),
            'model_type': 'UniversalTransformerModel',
            'model_architecture': 'UniversalTransformerModel',
            'best_val_loss': 0.0,
            'final_val_loss': 0.0,
            'epochs_trained': 0,
            'training_time': 'Unknown',
            'device': config.DEVICE,
            'model_config': config.MODEL_CONFIG.copy(),
            'train_config': config.TRAIN_CONFIG.copy(),
            'feature_count': len(feature_columns),
            'ticker_count': len(tickers),
            'tickers': tickers,
            'feature_columns': feature_columns,
            'target_feature': config.TARGET_FEATURE,
            'accuracy_metrics': {},
            'data_path': config.DATA_PATH,
            'version': '1.0'
        }
        
        metadata_file = model_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=self._json_serializer)
    
    def validate_all_models(self):
        models = []
        for model_dir in self.saved_models_path.iterdir():
            if model_dir.is_dir() and model_dir.name.startswith("model_"):
                if self.validate_and_migrate_model(model_dir):
                    if self._validate_model_files(model_dir):
                        models.append(model_dir)
                    else:
                        print(f"{Fore.YELLOW}Model {model_dir.name} has missing files{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Failed to validate model {model_dir.name}{Style.RESET_ALL}")
        
        print(f"{Fore.GREEN}Validated {len(models)} models{Style.RESET_ALL}")
        return models
