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
    
    try:
        data = pd.read_parquet(config.DATA_PATH)
        engine = UniversalModelEngine(data)
        model_folder = engine.train()
        
        if not model_folder:
            print(f"{Fore.RED}Training failed - no model folder returned{Style.RESET_ALL}")
            return
        
        manager = ModelManager()
        
        print(f"{Fore.CYAN}Verifying model integrity...{Style.RESET_ALL}")
        if not manager.verify_model_integrity(model_folder):
            print(f"{Fore.RED}Model integrity check failed{Style.RESET_ALL}")
            return
            
        print(f"{Fore.CYAN}Running accuracy testing...{Style.RESET_ALL}")
        analysis_files, results = test_accuracy.main()
        
        if model_folder and analysis_files and results:
            manager.copy_accuracy_analysis(model_folder, analysis_files)
            manager.update_model_accuracy(model_folder, results['accuracy_metrics'])
            print(f"{Fore.GREEN}Updated model accuracy metrics{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Model and analysis saved successfully{Style.RESET_ALL}")
        
        print(f"{Fore.CYAN}Creating additional backup...{Style.RESET_ALL}")
        backup_path = manager.create_model_backup(model_folder, "compressed")
        if backup_path:
            print(f"{Fore.GREEN}Compressed backup created: {backup_path}{Style.RESET_ALL}")
        
        print(f"{Fore.CYAN}Running automatic maintenance...{Style.RESET_ALL}")
        manager.cleanup_old_models(keep_count=5)
        manager.cleanup_old_backups(keep_count=15, keep_days=45)
        
        manager.display_model_summary()
        
    except FileNotFoundError as e:
        print(f"{Fore.RED}Data file not found: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Please ensure data is available at: {config.DATA_PATH}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Training failed with error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()