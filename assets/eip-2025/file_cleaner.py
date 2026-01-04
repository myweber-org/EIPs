import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

class TemporaryFileCleaner:
    def __init__(self, target_dir: Optional[str] = None):
        self.target_dir = Path(target_dir) if target_dir else Path.cwd()
        self.temp_extensions = {'.tmp', '.temp', '.bak', '.swp', '.log'}
        
    def identify_temp_files(self) -> List[Path]:
        temp_files = []
        for file_path in self.target_dir.rglob('*'):
            if file_path.is_file():
                if file_path.suffix.lower() in self.temp_extensions:
                    temp_files.append(file_path)
                elif file_path.name.startswith('~'):
                    temp_files.append(file_path)
        return temp_files
    
    def calculate_space(self, files: List[Path]) -> int:
        total_size = 0
        for file_path in files:
            try:
                total_size += file_path.stat().st_size
            except OSError:
                continue
        return total_size
    
    def clean_files(self, files: List[Path], dry_run: bool = False) -> dict:
        results = {
            'deleted': 0,
            'failed': 0,
            'space_freed': 0
        }
        
        for file_path in files:
            try:
                file_size = file_path.stat().st_size
                if not dry_run:
                    file_path.unlink()
                    results['deleted'] += 1
                    results['space_freed'] += file_size
                else:
                    results['deleted'] += 1
                    results['space_freed'] += file_size
            except OSError as e:
                results['failed'] += 1
                if not dry_run:
                    print(f"Failed to delete {file_path}: {e}")
        
        return results
    
    def run_cleanup(self, dry_run: bool = False) -> dict:
        print(f"Scanning directory: {self.target_dir}")
        temp_files = self.identify_temp_files()
        
        if not temp_files:
            print("No temporary files found.")
            return {'deleted': 0, 'failed': 0, 'space_freed': 0}
        
        print(f"Found {len(temp_files)} temporary files.")
        total_space = self.calculate_space(temp_files)
        print(f"Total space used: {total_space / (1024*1024):.2f} MB")
        
        if dry_run:
            print("Dry run mode - no files will be deleted.")
        
        results = self.clean_files(temp_files, dry_run)
        
        if not dry_run:
            print(f"Deleted {results['deleted']} files.")
            print(f"Freed {results['space_freed'] / (1024*1024):.2f} MB of space.")
            if results['failed'] > 0:
                print(f"Failed to delete {results['failed']} files.")
        
        return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean temporary files from a directory.')
    parser.add_argument('--directory', '-d', help='Target directory to clean')
    parser.add_argument('--dry-run', action='store_true', help='Simulate cleanup without deleting')
    parser.add_argument('--extensions', nargs='+', help='Additional file extensions to clean')
    
    args = parser.parse_args()
    
    cleaner = TemporaryFileCleaner(args.directory)
    
    if args.extensions:
        cleaner.temp_extensions.update(args.extensions)
    
    results = cleaner.run_cleanup(dry_run=args.dry_run)
    
    if args.dry_run:
        print(f"Dry run complete. Would delete {results['deleted']} files.")

if __name__ == '__main__':
    main()