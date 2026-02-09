
import os
import shutil
import sys

def clean_directory(directory, extensions_to_remove=None, dry_run=False):
    """
    Remove files with specified extensions from a directory.
    If extensions_to_remove is None, default temporary extensions are used.
    """
    if extensions_to_remove is None:
        extensions_to_remove = {'.tmp', '.temp', '.log', '.bak', '~'}
    
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return False
    
    removed_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1]
            
            if file_ext in extensions_to_remove or file.endswith('~'):
                if dry_run:
                    print(f"[DRY RUN] Would remove: {file_path}")
                else:
                    try:
                        os.remove(file_path)
                        removed_files.append(file_path)
                        print(f"Removed: {file_path}")
                    except Exception as e:
                        print(f"Error removing {file_path}: {e}")
    
    if not dry_run:
        print(f"\nCleaning complete. Removed {len(removed_files)} files.")
    
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python file_cleaner.py <directory> [--dry-run]")
        print("Extensions to remove: .tmp, .temp, .log, .bak, ~")
        sys.exit(1)
    
    directory = sys.argv[1]
    dry_run = '--dry-run' in sys.argv
    
    clean_directory(directory, dry_run=dry_run)

if __name__ == "__main__":
    main()