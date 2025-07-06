import os
import json
import shutil
import datetime
import torch
import joblib
from pathlib import Path
from colorama import Fore, Style
import pandas as pd

try:
    from . import config
except (ImportError, ValueError):
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src import config

class ModelManager:
    def __init__(self):
        self.saved_models_path = Path(config.SAVED_MODELS_PATH)
        self.saved_models_path.mkdir(exist_ok=True)
        
    def save_completed_model(self, model_path, scaler_path, metadata):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_folder = self.saved_models_path / f"model_{timestamp}"
        model_folder.mkdir(exist_ok=True)
        
        shutil.copy2(model_path, model_folder / "model.pth")
        shutil.copy2(scaler_path, model_folder / "scaler.pkl")
        
        metadata_path = model_folder / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"{Fore.GREEN}Model saved to: {model_folder}{Style.RESET_ALL}")
        return str(model_folder)
    
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
                metadata_path = model_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    models.append({
                        'path': str(model_dir),
                        'name': model_dir.name,
                        'metadata': metadata
                    })
        return sorted(models, key=lambda x: x['name'], reverse=True)
    
    def load_model_for_prediction(self, model_folder):
        model_path = Path(model_folder)
        model_file = model_path / "model.pth"
        scaler_file = model_path / "scaler.pkl"
        
        if not model_file.exists() or not scaler_file.exists():
            raise FileNotFoundError(f"Model files not found in {model_folder}")
        
        return str(model_file), str(scaler_file)
    
    def update_model_accuracy(self, model_folder, accuracy_metrics):
        metadata_path = Path(model_folder) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata['accuracy_metrics'] = accuracy_metrics
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"{Fore.GREEN}Updated model accuracy metrics{Style.RESET_ALL}")
    
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
