import os
import sys
import argparse
from pathlib import Path

def clean_directory(directory, extensions, dry_run=False):
    """
    Remove files with specified extensions from the given directory.
    """
    target_dir = Path(directory)
    if not target_dir.exists():
        print(f"Directory does not exist: {directory}")
        return False
    
    if not target_dir.is_dir():
        print(f"Path is not a directory: {directory}")
        return False
    
    removed_count = 0
    for ext in extensions:
        pattern = f"*.{ext.lstrip('.')}"
        for file_path in target_dir.glob(pattern):
            if dry_run:
                print(f"[DRY RUN] Would remove: {file_path}")
            else:
                try:
                    file_path.unlink()
                    print(f"Removed: {file_path}")
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
    
    if dry_run:
        print(f"[DRY RUN] Would remove {removed_count} files")
    else:
        print(f"Successfully removed {removed_count} files")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Clean temporary files from a directory")
    parser.add_argument("directory", help="Directory to clean")
    parser.add_argument("-e", "--extensions", nargs="+", default=["tmp", "temp", "log", "bak"],
                       help="File extensions to remove (default: tmp temp log bak)")
    parser.add_argument("-d", "--dry-run", action="store_true",
                       help="Show what would be removed without actually deleting")
    
    args = parser.parse_args()
    
    clean_directory(args.directory, args.extensions, args.dry_run)

if __name__ == "__main__":
    main()