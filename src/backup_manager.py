import os
import json
import shutil
import datetime
import torch
import joblib
import time
import zipfile
import hashlib
from pathlib import Path
from colorama import Fore, Style
import pandas as pd
import numpy as np

try:
    from . import config
except (ImportError, ValueError):
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src import config

class BackupManager:
    def __init__(self):
        self.backup_root = Path(config.SAVED_MODELS_PATH) / "backups"
        self.incremental_root = Path(config.SAVED_MODELS_PATH) / "incremental"
        self.recovery_log = self.backup_root / "recovery.log"
        
        self.backup_root.mkdir(exist_ok=True, parents=True)
        self.incremental_root.mkdir(exist_ok=True, parents=True)
        
    def create_full_backup(self, model_folder, backup_name=None):
        if backup_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"full_backup_{timestamp}"
        
        backup_path = self.backup_root / backup_name
        model_path = Path(model_folder)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model folder not found: {model_folder}")
        
        try:
            shutil.copytree(model_path, backup_path, dirs_exist_ok=True)
            
            backup_metadata = {
                'backup_type': 'full',
                'created_at': datetime.datetime.now().isoformat(),
                'source_model': str(model_path),
                'backup_size': self._calculate_dir_size(backup_path),
                'model_hash': self._calculate_model_hash(backup_path / "model.pth"),
                'files_count': len(list(backup_path.rglob('*')))
            }
            
            with open(backup_path / "backup_metadata.json", 'w') as f:
                json.dump(backup_metadata, f, indent=2)
            
            self._log_operation("BACKUP", f"Full backup created: {backup_name}")
            print(f"{Fore.GREEN}Full backup created: {backup_path}{Style.RESET_ALL}")
            return str(backup_path)
            
        except Exception as e:
            if backup_path.exists():
                shutil.rmtree(backup_path)
            raise Exception(f"Failed to create backup: {str(e)}")
    
    def create_incremental_backup(self, model_folder, reference_backup=None):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"incremental_backup_{timestamp}"
        backup_path = self.incremental_root / backup_name
        backup_path.mkdir(exist_ok=True)
        
        model_path = Path(model_folder)
        
        if reference_backup is None:
            reference_backup = self._get_latest_backup()
        
        if reference_backup is None:
            return self.create_full_backup(model_folder, f"incremental_to_full_{timestamp}")
        
        try:
            changed_files = self._find_changed_files(model_path, Path(reference_backup))
            
            backup_metadata = {
                'backup_type': 'incremental',
                'created_at': datetime.datetime.now().isoformat(),
                'source_model': str(model_path),
                'reference_backup': str(reference_backup),
                'changed_files': []
            }
            
            for file_path in changed_files:
                rel_path = file_path.relative_to(model_path)
                dest_path = backup_path / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, dest_path)
                backup_metadata['changed_files'].append(str(rel_path))
            
            backup_metadata['backup_size'] = self._calculate_dir_size(backup_path)
            backup_metadata['files_count'] = len(changed_files)
            
            with open(backup_path / "backup_metadata.json", 'w') as f:
                json.dump(backup_metadata, f, indent=2)
            
            self._log_operation("INCREMENTAL_BACKUP", f"Incremental backup created: {backup_name}")
            print(f"{Fore.GREEN}Incremental backup created: {backup_path} ({len(changed_files)} files){Style.RESET_ALL}")
            return str(backup_path)
            
        except Exception as e:
            if backup_path.exists():
                shutil.rmtree(backup_path)
            raise Exception(f"Failed to create incremental backup: {str(e)}")
    
    def create_compressed_backup(self, model_folder, backup_name=None):
        if backup_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"compressed_backup_{timestamp}"
        
        backup_path = self.backup_root / f"{backup_name}.zip"
        model_path = Path(model_folder)
        
        try:
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in model_path.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(model_path)
                        zipf.write(file_path, arcname)
            
            backup_metadata = {
                'backup_type': 'compressed',
                'created_at': datetime.datetime.now().isoformat(),
                'source_model': str(model_path),
                'backup_size': backup_path.stat().st_size,
                'compression_ratio': backup_path.stat().st_size / self._calculate_dir_size(model_path)
            }
            
            metadata_path = backup_path.with_suffix('.metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(backup_metadata, f, indent=2)
            
            self._log_operation("COMPRESSED_BACKUP", f"Compressed backup created: {backup_name}")
            print(f"{Fore.GREEN}Compressed backup created: {backup_path}{Style.RESET_ALL}")
            return str(backup_path)
            
        except Exception as e:
            if backup_path.exists():
                backup_path.unlink()
            raise Exception(f"Failed to create compressed backup: {str(e)}")
    
    def restore_from_backup(self, backup_path, restore_location=None, force=False):
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        
        if restore_location is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            restore_location = Path(config.SAVED_MODELS_PATH) / f"restored_model_{timestamp}"
        else:
            restore_location = Path(restore_location)
        
        if restore_location.exists() and not force:
            raise FileExistsError(f"Restore location exists. Use force=True to overwrite: {restore_location}")
        
        try:
            if backup_path.suffix == '.zip':
                return self._restore_compressed_backup(backup_path, restore_location)
            else:
                return self._restore_directory_backup(backup_path, restore_location)
                
        except Exception as e:
            raise Exception(f"Failed to restore from backup: {str(e)}")
    
    def _restore_compressed_backup(self, backup_path, restore_location):
        restore_location.mkdir(exist_ok=True, parents=True)
        
        with zipfile.ZipFile(backup_path, 'r') as zipf:
            zipf.extractall(restore_location)
        
        self._log_operation("RESTORE", f"Restored from compressed backup: {backup_path} -> {restore_location}")
        print(f"{Fore.GREEN}Model restored from compressed backup: {restore_location}{Style.RESET_ALL}")
        return str(restore_location)
    
    def _restore_directory_backup(self, backup_path, restore_location):
        if restore_location.exists():
            shutil.rmtree(restore_location)
        
        shutil.copytree(backup_path, restore_location)
        
        if (backup_path / "backup_metadata.json").exists():
            (restore_location / "backup_metadata.json").unlink()
        
        self._log_operation("RESTORE", f"Restored from backup: {backup_path} -> {restore_location}")
        print(f"{Fore.GREEN}Model restored from backup: {restore_location}{Style.RESET_ALL}")
        return str(restore_location)
    
    def auto_backup_strategy(self, model_folder, strategy="smart"):
        if strategy == "smart":
            latest_backup = self._get_latest_backup()
            if latest_backup is None:
                return self.create_full_backup(model_folder)
            
            backup_age = self._get_backup_age(latest_backup)
            if backup_age > 7:
                return self.create_full_backup(model_folder)
            else:
                return self.create_incremental_backup(model_folder)
        
        elif strategy == "full":
            return self.create_full_backup(model_folder)
        
        elif strategy == "incremental":
            return self.create_incremental_backup(model_folder)
        
        elif strategy == "compressed":
            return self.create_compressed_backup(model_folder)
        
        else:
            raise ValueError(f"Unknown backup strategy: {strategy}")
    
    def verify_backup_integrity(self, backup_path):
        backup_path = Path(backup_path)
        
        if backup_path.suffix == '.zip':
            return self._verify_compressed_backup(backup_path)
        else:
            return self._verify_directory_backup(backup_path)
    
    def _verify_compressed_backup(self, backup_path):
        try:
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                bad_files = zipf.testzip()
                if bad_files is None:
                    print(f"{Fore.GREEN}Compressed backup integrity verified: {backup_path}{Style.RESET_ALL}")
                    return True
                else:
                    print(f"{Fore.RED}Corrupted files found in backup: {bad_files}{Style.RESET_ALL}")
                    return False
        except Exception as e:
            print(f"{Fore.RED}Failed to verify compressed backup: {str(e)}{Style.RESET_ALL}")
            return False
    
    def _verify_directory_backup(self, backup_path):
        metadata_path = backup_path / "backup_metadata.json"
        
        if not metadata_path.exists():
            print(f"{Fore.YELLOW}No backup metadata found, checking basic structure{Style.RESET_ALL}")
            required_files = ['model.pth', 'scaler.pkl', 'metadata.json']
            for file_name in required_files:
                if not (backup_path / file_name).exists():
                    print(f"{Fore.RED}Missing required file: {file_name}{Style.RESET_ALL}")
                    return False
            return True
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            if 'model_hash' in metadata:
                current_hash = self._calculate_model_hash(backup_path / "model.pth")
                if current_hash != metadata['model_hash']:
                    print(f"{Fore.RED}Model hash mismatch in backup{Style.RESET_ALL}")
                    return False
            
            print(f"{Fore.GREEN}Backup integrity verified: {backup_path}{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Failed to verify backup: {str(e)}{Style.RESET_ALL}")
            return False
    
    def list_backups(self):
        backups = []
        
        for backup_path in self.backup_root.iterdir():
            if backup_path.is_dir():
                metadata_path = backup_path / "backup_metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        backups.append({
                            'path': str(backup_path),
                            'name': backup_path.name,
                            'type': metadata.get('backup_type', 'unknown'),
                            'created_at': metadata.get('created_at', 'unknown'),
                            'size': metadata.get('backup_size', 0)
                        })
                    except:
                        continue
            elif backup_path.suffix == '.zip':
                metadata_path = backup_path.with_suffix('.metadata.json')
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        backups.append({
                            'path': str(backup_path),
                            'name': backup_path.name,
                            'type': metadata.get('backup_type', 'compressed'),
                            'created_at': metadata.get('created_at', 'unknown'),
                            'size': metadata.get('backup_size', 0)
                        })
                    except:
                        continue
        
        return sorted(backups, key=lambda x: x['created_at'], reverse=True)
    
    def cleanup_old_backups(self, keep_count=10, keep_days=30):
        backups = self.list_backups()
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=keep_days)
        
        to_delete = []
        kept_backups = 0
        
        for backup in backups:
            try:
                backup_date = datetime.datetime.fromisoformat(backup['created_at'])
                if backup_date < cutoff_date and kept_backups >= keep_count:
                    to_delete.append(backup['path'])
                else:
                    kept_backups += 1
            except:
                continue
        
        for backup_path in to_delete:
            try:
                backup_path = Path(backup_path)
                if backup_path.is_dir():
                    shutil.rmtree(backup_path)
                else:
                    backup_path.unlink()
                    metadata_path = backup_path.with_suffix('.metadata.json')
                    if metadata_path.exists():
                        metadata_path.unlink()
                
                self._log_operation("CLEANUP", f"Deleted old backup: {backup_path}")
                print(f"{Fore.YELLOW}Deleted old backup: {backup_path.name}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Failed to delete backup {backup_path}: {str(e)}{Style.RESET_ALL}")
        
        print(f"{Fore.GREEN}Cleanup complete. Deleted {len(to_delete)} old backups{Style.RESET_ALL}")
    
    def _get_latest_backup(self):
        backups = self.list_backups()
        if backups:
            return backups[0]['path']
        return None
    
    def _get_backup_age(self, backup_path):
        backup_path = Path(backup_path)
        metadata_path = backup_path / "backup_metadata.json"
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                backup_date = datetime.datetime.fromisoformat(metadata['created_at'])
                return (datetime.datetime.now() - backup_date).days
            except:
                pass
        
        return backup_path.stat().st_mtime
    
    def _find_changed_files(self, current_path, reference_path):
        changed_files = []
        
        for file_path in current_path.rglob('*'):
            if file_path.is_file() and file_path.name != "backup_metadata.json":
                rel_path = file_path.relative_to(current_path)
                ref_file = reference_path / rel_path
                
                if not ref_file.exists():
                    changed_files.append(file_path)
                elif self._files_differ(file_path, ref_file):
                    changed_files.append(file_path)
        
        return changed_files
    
    def _files_differ(self, file1, file2):
        return file1.stat().st_mtime != file2.stat().st_mtime or file1.stat().st_size != file2.stat().st_size
    
    def _calculate_dir_size(self, path):
        total_size = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def _calculate_model_hash(self, model_path):
        if not model_path.exists():
            return None
        
        hash_md5 = hashlib.md5()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _log_operation(self, operation_type, message):
        try:
            timestamp = datetime.datetime.now().isoformat()
            log_entry = f"[{timestamp}] {operation_type}: {message}\n"
            
            with open(self.recovery_log, 'a') as f:
                f.write(log_entry)
        except:
            pass
