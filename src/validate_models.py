import sys
import os
from colorama import Fore, Style

try:
    from .model_manager import ModelManager
    from .backup_manager import BackupManager
    from . import config
except (ImportError, ValueError):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.model_manager import ModelManager
    from src.backup_manager import BackupManager
    from src import config

def main():
    print(f"{Fore.CYAN}Comprehensive Model and Backup Validation{Style.RESET_ALL}")
    print("=" * 60)
    
    manager = ModelManager()
    backup_manager = BackupManager()
    
    print(f"\n{Fore.CYAN}Step 1: Validating and migrating existing models...{Style.RESET_ALL}")
    validated_models = manager.validate_all_models()
    print(f"{Fore.GREEN}Found {len(validated_models)} valid models{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}Step 2: Verifying model integrity...{Style.RESET_ALL}")
    valid_models = 0
    for model_dir in validated_models:
        if manager.verify_model_integrity(str(model_dir)):
            valid_models += 1
            print(f"{Fore.GREEN}{model_dir.name}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}{model_dir.name} - FAILED{Style.RESET_ALL}")
    
    print(f"{Fore.CYAN}Model integrity: {valid_models}/{len(validated_models)} passed{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}Step 3: Checking backups...{Style.RESET_ALL}")
    backups = backup_manager.list_backups()
    valid_backups = 0
    
    for backup in backups:
        if backup_manager.verify_backup_integrity(backup['path']):
            valid_backups += 1
            print(f"{Fore.GREEN}{backup['name']}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}{backup['name']} - FAILED{Style.RESET_ALL}")
    
    print(f"{Fore.CYAN}Backup integrity: {valid_backups}/{len(backups)} passed{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}Step 4: Creating missing backups...{Style.RESET_ALL}")
    for model_dir in validated_models:
        try:
            backup_path = manager.create_model_backup(str(model_dir), "smart")
            if backup_path:
                print(f"{Fore.GREEN}Backup created for {model_dir.name}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}Failed to backup {model_dir.name}: {str(e)}{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}Step 5: Running maintenance...{Style.RESET_ALL}")
    manager.cleanup_old_models(keep_count=5)
    manager.cleanup_old_backups(keep_count=15, keep_days=45)
    
    print(f"\n{Fore.CYAN}Step 6: Final summary...{Style.RESET_ALL}")
    manager.display_model_summary()
    
    print(f"\n{Fore.GREEN}Validation complete!{Style.RESET_ALL}")
    print(f"Valid models: {valid_models}")
    print(f"Valid backups: {valid_backups}")

if __name__ == "__main__":
    main()
