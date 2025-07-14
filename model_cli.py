#!/usr/bin/env python3

import argparse
import sys
import os
import subprocess
from colorama import Fore, Style

def train_model():
    print(f"{Fore.CYAN}Starting model training...{Style.RESET_ALL}")
    try:
        result = subprocess.run([sys.executable, "-c", 
                               "import sys; sys.path.append('src'); from src import train; train.main()"], 
                              cwd=os.path.dirname(__file__), capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"{Fore.RED}Training failed: {e}{Style.RESET_ALL}")
        return False

def test_model():
    print(f"{Fore.CYAN}Testing model accuracy...{Style.RESET_ALL}")
    try:
        result = subprocess.run([sys.executable, "-c", 
                               "import sys; sys.path.append('src'); from src import test_accuracy; test_accuracy.main()"], 
                              cwd=os.path.dirname(__file__), capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"{Fore.RED}Testing failed: {e}{Style.RESET_ALL}")
        return False

def run_app():
    print(f"{Fore.CYAN}Launching Streamlit app...{Style.RESET_ALL}")
    subprocess.run(["streamlit", "run", "src/app.py"], cwd=os.path.dirname(__file__))

def validate_models():
    print(f"{Fore.CYAN}Validating models...{Style.RESET_ALL}")
    try:
        result = subprocess.run([sys.executable, "-c", 
                               "import sys; sys.path.append('src'); from src import validate_models; validate_models.main()"], 
                              cwd=os.path.dirname(__file__), capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"{Fore.RED}Validation failed: {e}{Style.RESET_ALL}")
        return False

def visualize_predictions():
    print(f"{Fore.CYAN}Generating visualizations...{Style.RESET_ALL}")
    try:
        result = subprocess.run([sys.executable, "-c", 
                               "import sys; sys.path.append('src'); from src import visualize_predictions; visualize_predictions.main()"], 
                              cwd=os.path.dirname(__file__), capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"{Fore.RED}Visualization failed: {e}{Style.RESET_ALL}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Stock Prediction Model CLI")
    parser.add_argument('command', choices=['train', 'test', 'app', 'validate', 'visualize'], 
                       help='Command to execute')
    
    args = parser.parse_args()
    
    print(f"{Fore.GREEN}Stock Prediction Model CLI{Style.RESET_ALL}")
    print("=" * 40)
    
    if args.command == 'train':
        train_model()
    elif args.command == 'test':
        test_model()
    elif args.command == 'app':
        run_app()
    elif args.command == 'validate':
        validate_models()
    elif args.command == 'visualize':
        visualize_predictions()

if __name__ == "__main__":
    main()
