import torch
from colorama import Fore, Style, init
import os

init(autoreset=True)

RAW_DATA_CACHE_PATH = "data/raw_cache/"
DATA_PATH = "data/universal_stock_data.parquet"
MODELS_PATH = "models/"
SAVED_MODELS_PATH = "saved_models/"
CHECKPOINT_PATH = "models/checkpoint.pth"

def get_device():
    if torch.backends.mps.is_available():
        print(f"{Fore.GREEN}Apple Silicon GPU (MPS) detected and available{Style.RESET_ALL}")
        return "mps"
    elif torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"{Fore.GREEN}Multiple NVIDIA GPUs detected: {gpu_count} GPUs available{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}NVIDIA GPU (CUDA) detected and available{Style.RESET_ALL}")
        return "cuda"
    else:
        print(f"{Fore.YELLOW}Using CPU - no GPU acceleration available{Style.RESET_ALL}")
        return "cpu"

DEVICE = get_device()
USE_MULTI_GPU = torch.cuda.device_count() > 1 if DEVICE == "cuda" else False

if DEVICE == "mps":
    torch.backends.mps.enable_nested_tensor = False
    torch.mps.set_per_process_memory_fraction(0.8)

START_DATE = "2011-01-01"
TICKERS = ["AAPL", "MSFT", "GOOG", "NVDA", "TSLA", "AMZN", "META"]

YFINANCE_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']

FRED_SERIES = {
    "interest_rate": "FEDFUNDS",
    "gdp_growth": "GDPC1",
    "inflation": "CPIAUCSL",
    "unemployment": "UNRATE",
    "vix": "VIXCLS"
}

MACRO_FEATURES = list(FRED_SERIES.keys())

TECHNICAL_INDICATORS = [
    {"kind": "rsi", "length": 14},
    {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
    {"kind": "bbands", "length": 20, "std": 2},
    {"kind": "atr", "length": 14},
    {"kind": "stoch", "k": 14, "d": 3},
    {"kind": "williams", "length": 14},
    {"kind": "adx", "length": 14},
    {"kind": "cmf", "length": 20},
    {"kind": "mfi", "length": 14},
    {"kind": "obv"},
]

MODEL_CONFIG = {
    "d_model": 384,
    "n_heads": 24,
    "n_layers": 16,
    "dropout": 0.25,
    "sequence_length": 60,
    "prediction_horizon": 5,
    "ticker_embedding_dim": 48,
    "use_layer_norm": True,
    "use_residual_connections": True,
    "activation": "gelu",
    "attention_dropout": 0.15,
    "ffn_dropout": 0.25,
    "ffn_dim": 1536
}

TRAIN_CONFIG = {
    "learning_rate": 0.00008,
    "max_lr": 0.0002,
    "weight_decay": 8e-06,
    "batch_size": 64,
    "epochs": 800,
    "val_split_ratio": 0.2,
    "lr_scheduler_patience": 10,
    "lr_scheduler_factor": 0.6,
    "warmup_epochs": 50,
    "gradient_clip_val": 0.8,
    "early_stopping_patience": 250,
    "mixed_precision": True,
    "accumulate_grad_batches": 4,
    "dataloader_num_workers": 4,
    "pin_memory": True,
    "target_val_loss": 0.0005,
    "checkpoint_every": 5,
    "save_best_only": True,
}

TARGET_FEATURE = 'log_return_Close'

BASE_FEATURE_COLUMNS = [
    'log_return_Open', 'log_return_High', 'log_return_Low', 'log_return_Close',
    'log_volume', 'RSI_14', 'MACDh_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0',
    'BBU_20_2.0', 'ATRr_14'
]

TECHNICAL_FEATURES = BASE_FEATURE_COLUMNS
TIME_FEATURES = ['day_of_week', 'day_of_month', 'week_of_year', 'month']
MACRO_FEATURES = list(FRED_SERIES.keys())

FEATURE_ENGINEERING = {
    "rolling_windows": [5, 10, 20, 50],
    "volatility_windows": [10, 20, 30],
    "momentum_periods": [5, 10, 20],
    "use_lagged_features": True,
    "lag_periods": [1, 2, 3, 5],
    "use_technical_patterns": True,
}

BACKUP_CONFIG = {
    "auto_backup": True,
    "backup_strategy": "smart",
    "max_backups": 15,
    "backup_retention_days": 45,
    "compression_level": 6,
    "verify_integrity": True,
    "create_checkpoint_every": 5,
    "emergency_backup_threshold": 0.1
}

RECOVERY_CONFIG = {
    "auto_repair": True,
    "repair_metadata": True,
    "backup_before_repair": True,
    "max_recovery_attempts": 3,
    "recovery_log_retention_days": 90
}

def ensure_directories():
    dirs = [RAW_DATA_CACHE_PATH, MODELS_PATH, SAVED_MODELS_PATH]
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"{Fore.GREEN}Created directory: {dir_path}{Style.RESET_ALL}")

ensure_directories()