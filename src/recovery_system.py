import os
import sys
import json
import datetime
from pathlib import Path
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

class ModelRecoverySystem:
    def __init__(self):
        self.model_manager = ModelManager()
        self.backup_manager = BackupManager()
        self.recovery_log_path = Path(config.SAVED_MODELS_PATH) / "recovery_operations.log"
    
    def emergency_recovery(self):
        print(f"{Fore.RED}EMERGENCY RECOVERY MODE ACTIVATED{Style.RESET_ALL}")
        print("=" * 50)
        
        recovery_count = 0
        
        print(f"\n{Fore.CYAN}Step 1: Scanning for recoverable models...{Style.RESET_ALL}")
        corrupted_models = self._find_corrupted_models()
        
        if not corrupted_models:
            print(f"{Fore.GREEN}No corrupted models found{Style.RESET_ALL}")
            return recovery_count
        
        print(f"{Fore.YELLOW}Found {len(corrupted_models)} corrupted models{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}Step 2: Attempting recovery from backups...{Style.RESET_ALL}")
        for model_name in corrupted_models:
            if self._recover_single_model(model_name):
                recovery_count += 1
                print(f"{Fore.GREEN}Recovered {model_name}{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Failed to recover {model_name}{Style.RESET_ALL}")
        
        self._log_recovery_operation("EMERGENCY_RECOVERY", f"Recovered {recovery_count}/{len(corrupted_models)} models")
        
        print(f"\n{Fore.CYAN}Recovery Summary:{Style.RESET_ALL}")
        print(f"Models recovered: {recovery_count}")
        print(f"Models failed: {len(corrupted_models) - recovery_count}")
        
        return recovery_count
    
    def selective_recovery(self, model_name, backup_date=None):
        print(f"{Fore.CYAN}Selective Recovery: {model_name}{Style.RESET_ALL}")
        
        available_backups = self._find_model_backups(model_name)
        
        if not available_backups:
            print(f"{Fore.RED}No backups found for model: {model_name}{Style.RESET_ALL}")
            return False
        
        if backup_date:
            target_backup = self._find_backup_by_date(available_backups, backup_date)
        else:
            target_backup = available_backups[0]
        
        if not target_backup:
            print(f"{Fore.RED}No suitable backup found{Style.RESET_ALL}")
            return False
        
        success = self._restore_from_specific_backup(target_backup, model_name)
        
        if success:
            self._log_recovery_operation("SELECTIVE_RECOVERY", f"Recovered {model_name} from {target_backup['path']}")
            print(f"{Fore.GREEN}Successfully recovered {model_name}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Failed to recover {model_name}{Style.RESET_ALL}")
        
        return success
    
    def point_in_time_recovery(self, target_date):
        print(f"{Fore.CYAN}Point-in-Time Recovery: {target_date}{Style.RESET_ALL}")
        
        target_datetime = datetime.datetime.fromisoformat(target_date)
        recovery_count = 0
        
        backups = self.backup_manager.list_backups()
        suitable_backups = []
        
        for backup in backups:
            try:
                backup_datetime = datetime.datetime.fromisoformat(backup['created_at'])
                if backup_datetime <= target_datetime:
                    suitable_backups.append(backup)
            except:
                continue
        
        suitable_backups.sort(key=lambda x: x['created_at'], reverse=True)
        
        if not suitable_backups:
            print(f"{Fore.RED}No backups found before target date{Style.RESET_ALL}")
            return recovery_count
        
        print(f"Found {len(suitable_backups)} suitable backups")
        
        for backup in suitable_backups[:5]:
            try:
                model_name = f"recovered_{target_date.replace(':', '-')}"
                restored_path = self.model_manager.restore_model_from_backup(backup['path'], model_name)
                if restored_path:
                    recovery_count += 1
                    print(f"{Fore.GREEN}Restored backup: {backup['name']}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.YELLOW}Failed to restore {backup['name']}: {str(e)}{Style.RESET_ALL}")
        
        self._log_recovery_operation("POINT_IN_TIME_RECOVERY", f"Recovered {recovery_count} models to {target_date}")
        return recovery_count
    
    def validate_and_repair(self):
        print(f"{Fore.CYAN}Model Validation and Repair{Style.RESET_ALL}")
        
        models = self.model_manager.list_saved_models()
        repaired_count = 0
        
        for model in models:
            model_path = model['path']
            
            if not self.model_manager.verify_model_integrity(model_path):
                print(f"{Fore.YELLOW}Repairing {model['name']}...{Style.RESET_ALL}")
                
                if self._repair_model_metadata(model_path):
                    repaired_count += 1
                    print(f"{Fore.GREEN}Repaired metadata for {model['name']}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Failed to repair {model['name']}{Style.RESET_ALL}")
        
        self._log_recovery_operation("VALIDATE_AND_REPAIR", f"Repaired {repaired_count} models")
        return repaired_count
    
    def create_recovery_checkpoint(self):
        print(f"{Fore.CYAN}Creating Recovery Checkpoint{Style.RESET_ALL}")
        
        checkpoint_name = f"recovery_checkpoint_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        models = self.model_manager.list_saved_models()
        
        checkpoint_count = 0
        
        for model in models:
            try:
                backup_path = self.backup_manager.create_full_backup(model['path'], f"{checkpoint_name}_{model['name']}")
                if backup_path:
                    checkpoint_count += 1
            except Exception as e:
                print(f"{Fore.YELLOW}Failed to backup {model['name']}: {str(e)}{Style.RESET_ALL}")
        
        self._log_recovery_operation("RECOVERY_CHECKPOINT", f"Created checkpoint with {checkpoint_count} model backups")
        print(f"{Fore.GREEN}Recovery checkpoint created with {checkpoint_count} backups{Style.RESET_ALL}")
        
        return checkpoint_count
    
    def _find_corrupted_models(self):
        models = self.model_manager.list_saved_models()
        corrupted = []
        
        for model in models:
            if not self.model_manager.verify_model_integrity(model['path']):
                corrupted.append(model['name'])
        
        return corrupted
    
    def _recover_single_model(self, model_name):
        backups = self._find_model_backups(model_name)
        
        for backup in backups:
            try:
                restored_path = self.model_manager.restore_model_from_backup(backup['path'], f"recovered_{model_name}")
                if restored_path and self.model_manager.verify_model_integrity(restored_path):
                    return True
            except:
                continue
        
        return False
    
    def _find_model_backups(self, model_name):
        backups = self.backup_manager.list_backups()
        model_backups = []
        
        for backup in backups:
            if model_name in backup['name']:
                model_backups.append(backup)
        
        return sorted(model_backups, key=lambda x: x['created_at'], reverse=True)
    
    def _find_backup_by_date(self, backups, target_date):
        target_datetime = datetime.datetime.fromisoformat(target_date)
        
        for backup in backups:
            try:
                backup_datetime = datetime.datetime.fromisoformat(backup['created_at'])
                if abs((backup_datetime - target_datetime).total_seconds()) < 3600:
                    return backup
            except:
                continue
        
        return None
    
    def _restore_from_specific_backup(self, backup, model_name):
        try:
            restored_path = self.model_manager.restore_model_from_backup(backup['path'], f"recovered_{model_name}")
            return restored_path is not None
        except:
            return False
    
    def _repair_model_metadata(self, model_path):
        try:
            metadata_path = Path(model_path) / "metadata.json"
            
            if not metadata_path.exists():
                self.model_manager._create_default_metadata(Path(model_path))
                return True
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            updated_metadata = self.model_manager._ensure_complete_metadata(metadata)
            
            with open(metadata_path, 'w') as f:
                json.dump(updated_metadata, f, indent=2, default=self.model_manager._json_serializer)
            
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Metadata repair failed: {str(e)}{Style.RESET_ALL}")
            return False
    
    def _log_recovery_operation(self, operation_type, message):
        try:
            timestamp = datetime.datetime.now().isoformat()
            log_entry = f"[{timestamp}] {operation_type}: {message}\n"
            
            with open(self.recovery_log_path, 'a') as f:
                f.write(log_entry)
        except:
            pass

def main():
    recovery_system = ModelRecoverySystem()
    
    import argparse
    parser = argparse.ArgumentParser(description='Model Recovery System')
    parser.add_argument('command', choices=['emergency', 'validate', 'checkpoint', 'selective', 'point-in-time'])
    parser.add_argument('--model', help='Model name for selective recovery')
    parser.add_argument('--date', help='Target date for point-in-time recovery (ISO format)')
    
    args = parser.parse_args()
    
    if args.command == 'emergency':
        recovery_system.emergency_recovery()
    elif args.command == 'validate':
        recovery_system.validate_and_repair()
    elif args.command == 'checkpoint':
        recovery_system.create_recovery_checkpoint()
    elif args.command == 'selective':
        if not args.model:
            print(f"{Fore.RED}Model name required for selective recovery{Style.RESET_ALL}")
            return
        recovery_system.selective_recovery(args.model, args.date)
    elif args.command == 'point-in-time':
        if not args.date:
            print(f"{Fore.RED}Target date required for point-in-time recovery{Style.RESET_ALL}")
            return
        recovery_system.point_in_time_recovery(args.date)

if __name__ == "__main__":
    main()
