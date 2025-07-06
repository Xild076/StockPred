import pandas as pd
import os
import sys
import json
from colorama import Fore, Style

try:
    from .modeling.engine import UniversalModelEngine
    from . import config
    from . import test_accuracy
    from .model_manager import ModelManager
except (ImportError, ValueError):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.modeling.engine import UniversalModelEngine
    from src import config
    from src import test_accuracy
    from src.model_manager import ModelManager

def main():
    print(f"{Fore.CYAN}Starting Enhanced Stock Prediction Training{Style.RESET_ALL}")
    data = pd.read_parquet(config.DATA_PATH)
    engine = UniversalModelEngine(data)
    model_folder = engine.train()
    print(f"{Fore.CYAN}Running accuracy testing...{Style.RESET_ALL}")
    analysis_files, results = test_accuracy.main()
    
    if model_folder and analysis_files and results:
        manager = ModelManager()
        manager.copy_accuracy_analysis(model_folder, analysis_files)
        manager.update_model_accuracy(model_folder, results['accuracy_metrics'])
        print(f"{Fore.GREEN}Updated model accuracy metrics{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Model and analysis saved successfully{Style.RESET_ALL}")
    
    manager = ModelManager()
    manager.display_model_summary()

if __name__ == "__main__":
    main()