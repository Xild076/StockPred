import argparse
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

def list_models():
    manager = ModelManager()
    models = manager.list_saved_models()
    
    if not models:
        print(f"{Fore.YELLOW}No saved models found{Style.RESET_ALL}")
        return
    
    print(f"{Fore.CYAN}Available Models:{Style.RESET_ALL}")
    print("-" * 80)
    
    for i, model in enumerate(models, 1):
        metadata = model['metadata']
        print(f"{Fore.WHITE}{i}. {model['name']}{Style.RESET_ALL}")
        print(f"   Created: {metadata.get('created_at', 'Unknown')}")
        print(f"   Val Loss: {metadata.get('best_val_loss', 'N/A')}")
        print(f"   Path: {model['path']}")
        print()

def list_backups():
    backup_manager = BackupManager()
    backups = backup_manager.list_backups()
    
    if not backups:
        print(f"{Fore.YELLOW}No backups found{Style.RESET_ALL}")
        return
    
    print(f"{Fore.CYAN}Available Backups:{Style.RESET_ALL}")
    print("-" * 80)
    
    for i, backup in enumerate(backups, 1):
        print(f"{Fore.WHITE}{i}. {backup['name']}{Style.RESET_ALL}")
        print(f"   Type: {backup['type']}")
        print(f"   Created: {backup['created_at']}")
        print(f"   Size: {backup['size']:,} bytes")
        print(f"   Path: {backup['path']}")
        print()

def create_backup(model_path, backup_type):
    if not os.path.exists(model_path):
        print(f"{Fore.RED}Model path not found: {model_path}{Style.RESET_ALL}")
        return
    
    manager = ModelManager()
    backup_path = manager.create_model_backup(model_path, backup_type)
    
    if backup_path:
        print(f"{Fore.GREEN}Backup created successfully: {backup_path}{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Failed to create backup{Style.RESET_ALL}")

def restore_backup(backup_path, restore_name=None):
    manager = ModelManager()
    restored_path = manager.restore_model_from_backup(backup_path, restore_name)
    
    if restored_path:
        print(f"{Fore.GREEN}Model restored successfully: {restored_path}{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Failed to restore model{Style.RESET_ALL}")

def verify_model(model_path):
    manager = ModelManager()
    is_valid = manager.verify_model_integrity(model_path)
    
    if is_valid:
        print(f"{Fore.GREEN}Model integrity verified{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Model integrity check failed{Style.RESET_ALL}")

def verify_backup(backup_path):
    backup_manager = BackupManager()
    is_valid = backup_manager.verify_backup_integrity(backup_path)
    
    if is_valid:
        print(f"{Fore.GREEN}Backup integrity verified{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Backup integrity check failed{Style.RESET_ALL}")

def cleanup_models(keep_count):
    manager = ModelManager()
    manager.cleanup_old_models(keep_count)

def cleanup_backups(keep_count, keep_days):
    manager = ModelManager()
    manager.cleanup_old_backups(keep_count, keep_days)

def auto_maintenance():
    print(f"{Fore.CYAN}Running automatic maintenance...{Style.RESET_ALL}")
    
    manager = ModelManager()
    
    print("Cleaning up old models...")
    manager.cleanup_old_models(keep_count=5)
    
    print("Cleaning up old backups...")
    manager.cleanup_old_backups(keep_count=10, keep_days=30)
    
    print("Verifying existing models...")
    models = manager.list_saved_models()
    valid_count = 0
    
    for model in models:
        if manager.verify_model_integrity(model['path']):
            valid_count += 1
    
    print(f"{Fore.GREEN}Maintenance complete. {valid_count}/{len(models)} models verified{Style.RESET_ALL}")

def main():
    parser = argparse.ArgumentParser(description='Model Backup and Recovery Management')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    subparsers.add_parser('list-models', help='List all saved models')
    subparsers.add_parser('list-backups', help='List all backups')
    
    backup_parser = subparsers.add_parser('create-backup', help='Create a backup of a model')
    backup_parser.add_argument('model_path', help='Path to the model folder')
    backup_parser.add_argument('--type', choices=['full', 'incremental', 'compressed', 'smart'], 
                              default='smart', help='Backup type')
    
    restore_parser = subparsers.add_parser('restore-backup', help='Restore a model from backup')
    restore_parser.add_argument('backup_path', help='Path to the backup')
    restore_parser.add_argument('--name', help='Name for the restored model')
    
    verify_model_parser = subparsers.add_parser('verify-model', help='Verify model integrity')
    verify_model_parser.add_argument('model_path', help='Path to the model folder')
    
    verify_backup_parser = subparsers.add_parser('verify-backup', help='Verify backup integrity')
    verify_backup_parser.add_argument('backup_path', help='Path to the backup')
    
    cleanup_models_parser = subparsers.add_parser('cleanup-models', help='Clean up old models')
    cleanup_models_parser.add_argument('--keep', type=int, default=5, help='Number of models to keep')
    
    cleanup_backups_parser = subparsers.add_parser('cleanup-backups', help='Clean up old backups')
    cleanup_backups_parser.add_argument('--keep-count', type=int, default=10, help='Number of backups to keep')
    cleanup_backups_parser.add_argument('--keep-days', type=int, default=30, help='Days to keep backups')
    
    subparsers.add_parser('auto-maintenance', help='Run automatic maintenance')
    
    args = parser.parse_args()
    
    if args.command == 'list-models':
        list_models()
    elif args.command == 'list-backups':
        list_backups()
    elif args.command == 'create-backup':
        create_backup(args.model_path, args.type)
    elif args.command == 'restore-backup':
        restore_backup(args.backup_path, args.name)
    elif args.command == 'verify-model':
        verify_model(args.model_path)
    elif args.command == 'verify-backup':
        verify_backup(args.backup_path)
    elif args.command == 'cleanup-models':
        cleanup_models(args.keep)
    elif args.command == 'cleanup-backups':
        cleanup_backups(args.keep_count, args.keep_days)
    elif args.command == 'auto-maintenance':
        auto_maintenance()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
