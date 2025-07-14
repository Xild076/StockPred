#!/usr/bin/env python3

import sys
import os
import pandas as pd
import torch
from colorama import Fore, Style, init

init(autoreset=True)

def check_python_environment():
    print(f"{Fore.CYAN}Checking Python Environment...{Style.RESET_ALL}")
    
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"{Fore.GREEN}Python version: {python_version}{Style.RESET_ALL}")
    
    dependencies = [
        ('torch', '2.7.1'),
        ('pandas', '2.3.0'),
        ('numpy', None),
        ('sklearn', None),
        ('streamlit', None),
        ('yfinance', None),
        ('pandas_ta', None),
        ('colorama', None)
    ]
    
    for dep, expected_version in dependencies:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            if expected_version and version != expected_version:
                print(f"{Fore.YELLOW}{dep}: {version} (expected {expected_version}){Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}{dep}: {version}{Style.RESET_ALL}")
        except ImportError:
            print(f"{Fore.RED}{dep}: NOT INSTALLED{Style.RESET_ALL}")
            return False
    
    return True

def check_hardware_support():
    print(f"\n{Fore.CYAN}Checking Hardware Support...{Style.RESET_ALL}")
    
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    
    print(f"{Fore.GREEN}CUDA available: {cuda_available}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}MPS (Apple Silicon) available: {mps_available}{Style.RESET_ALL}")
    
    if mps_available:
        print(f"{Fore.GREEN}Using Apple Silicon GPU acceleration{Style.RESET_ALL}")
    elif cuda_available:
        print(f"{Fore.GREEN}Using NVIDIA GPU acceleration{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}Using CPU only (no GPU acceleration){Style.RESET_ALL}")
    
    return True

def check_project_structure():
    print(f"\n{Fore.CYAN}Checking Project Structure...{Style.RESET_ALL}")
    
    required_dirs = [
        'src',
        'src/modeling',
        'src/data_processing',
        'data',
        'data/raw_cache',
        'models',
        'saved_models'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"{Fore.GREEN}{dir_path}/{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}{dir_path}/ - MISSING{Style.RESET_ALL}")
            return False
    
    return True

def check_data_availability():
    print(f"\n{Fore.CYAN}Checking Data Availability...{Style.RESET_ALL}")
    
    sys.path.insert(0, 'src')
    from src import config
    
    if os.path.exists(config.DATA_PATH):
        data = pd.read_parquet(config.DATA_PATH)
        print(f"{Fore.GREEN}Main data file: {data.shape}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Tickers available: {sorted(data['ticker'].unique())}{Style.RESET_ALL}")
        
        required_features = config.TECHNICAL_FEATURES + config.TIME_FEATURES + config.MACRO_FEATURES
        missing_features = [f for f in required_features if f not in data.columns]
        
        if missing_features:
            print(f"{Fore.RED}Missing features: {missing_features}{Style.RESET_ALL}")
            return False
        else:
            print(f"{Fore.GREEN}All {len(required_features)} required features present{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Main data file not found: {config.DATA_PATH}{Style.RESET_ALL}")
        return False
    
    cache_files = ['fred_raw.parquet', 'yfinance_raw.parquet']
    for cache_file in cache_files:
        cache_path = os.path.join(config.RAW_DATA_CACHE_PATH, cache_file)
        if os.path.exists(cache_path):
            print(f"{Fore.GREEN}Cache file: {cache_file}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}Cache file missing: {cache_file} (will be created on first run){Style.RESET_ALL}")
    
    return True

def check_model_imports():
    print(f"\n{Fore.CYAN}Checking Model Components...{Style.RESET_ALL}")
    
    try:
        sys.path.insert(0, 'src')
        
        from src import config
        print(f"{Fore.GREEN}Config module{Style.RESET_ALL}")
        
        from src.modeling.architecture import UniversalTransformerModel
        print(f"{Fore.GREEN}Model architecture{Style.RESET_ALL}")
        
        from src.modeling.engine import UniversalModelEngine
        print(f"{Fore.GREEN}Model engine{Style.RESET_ALL}")
        
        from src.model_manager import ModelManager
        print(f"{Fore.GREEN}Model manager{Style.RESET_ALL}")
        
        from src.backup_manager import BackupManager
        print(f"{Fore.GREEN}Backup manager{Style.RESET_ALL}")
        
        from src.recovery_system import ModelRecoverySystem
        print(f"{Fore.GREEN}Recovery system{Style.RESET_ALL}")
        
        from src.data_processing import processor
        print(f"{Fore.GREEN}Data processor{Style.RESET_ALL}")
        
        data = pd.read_parquet(config.DATA_PATH)
        engine = UniversalModelEngine(data)
        print(f"{Fore.GREEN}Model engine instantiation{Style.RESET_ALL}")
        
        return True
        
    except Exception as e:
        print(f"{Fore.RED}Import error: {e}{Style.RESET_ALL}")
        return False

def check_saved_models():
    print(f"\n{Fore.CYAN}Checking Saved Models...{Style.RESET_ALL}")
    
    sys.path.insert(0, 'src')
    from src import config
    
    saved_models_path = config.SAVED_MODELS_PATH
    
    if not os.path.exists(saved_models_path):
        print(f"{Fore.YELLOW}No saved models directory found{Style.RESET_ALL}")
        return True
    
    model_folders = [d for d in os.listdir(saved_models_path) 
                    if os.path.isdir(os.path.join(saved_models_path, d))]
    
    if not model_folders:
        print(f"{Fore.YELLOW}No saved models found (run training to create models){Style.RESET_ALL}")
        return True
    
    print(f"{Fore.GREEN}Found {len(model_folders)} saved model(s){Style.RESET_ALL}")
    
    for model_folder in model_folders:
        model_path = os.path.join(saved_models_path, model_folder)
        required_files = ['model.pth', 'scaler.pkl', 'metadata.json']
        
        folder_contents = os.listdir(model_path)
        if not folder_contents:
            print(f"{Fore.YELLOW}{model_folder}: empty folder (will be cleaned up){Style.RESET_ALL}")
            continue
        
        all_present = True
        for req_file in required_files:
            file_path = os.path.join(model_path, req_file)
            if not os.path.exists(file_path):
                print(f"{Fore.RED}{model_folder}: missing {req_file}{Style.RESET_ALL}")
                all_present = False
        
        if all_present:
            print(f"{Fore.GREEN}{model_folder}: complete{Style.RESET_ALL}")
    
    return True

def run_full_system_check():
    print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Stock Prediction Model - System Health Check{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
    
    checks = [
        ("Python Environment", check_python_environment),
        ("Hardware Support", check_hardware_support),
        ("Project Structure", check_project_structure),
        ("Data Availability", check_data_availability),
        ("Model Components", check_model_imports),
        ("Saved Models", check_saved_models)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"{Fore.RED}{check_name}: FAILED with error: {e}{Style.RESET_ALL}")
            all_passed = False
    
    print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
    
    if all_passed:
        print(f"{Fore.GREEN}ALL CHECKS PASSED - System is ready!{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}You can now:{Style.RESET_ALL}")
        print(f"  • Run training: python model_cli.py train")
        print(f"  • Launch app: python model_cli.py app")
        print(f"  • Test accuracy: python model_cli.py test")
        print(f"  • Validate models: python model_cli.py validate")
    else:
        print(f"{Fore.RED}SOME CHECKS FAILED - Please fix issues above{Style.RESET_ALL}")
    
    print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
    
    return all_passed

if __name__ == "__main__":
    run_full_system_check()
