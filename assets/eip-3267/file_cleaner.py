
import os
import shutil
import tempfile
from pathlib import Path

def clean_temp_files(directory: str, extensions: tuple = ('.tmp', '.temp', '.log'), max_age_days: int = 7):
    """
    Remove temporary files with specified extensions older than a given number of days.
    
    Args:
        directory: Path to the directory to clean.
        extensions: Tuple of file extensions to consider as temporary.
        max_age_days: Maximum age of files in days before they are deleted.
    """
    target_dir = Path(directory)
    if not target_dir.exists() or not target_dir.is_dir():
        raise ValueError(f"Invalid directory: {directory}")

    current_time = time.time()
    cutoff_time = current_time - (max_age_days * 24 * 60 * 60)

    for item in target_dir.rglob('*'):
        if item.is_file() and item.suffix.lower() in extensions:
            if item.stat().st_mtime < cutoff_time:
                try:
                    item.unlink()
                    print(f"Deleted: {item}")
                except OSError as e:
                    print(f"Error deleting {item}: {e}")
        elif item.is_dir() and item.name.startswith('tmp_'):
            try:
                shutil.rmtree(item)
                print(f"Removed directory: {item}")
            except OSError as e:
                print(f"Error removing directory {item}: {e}")

if __name__ == "__main__":
    import time
    temp_dir = tempfile.gettempdir()
    print(f"Cleaning temporary files in: {temp_dir}")
    clean_temp_files(temp_dir)import sys
import os

def remove_duplicates(input_file, output_file=None):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return False
    
    if output_file is None:
        output_file = input_file + ".deduped"
    
    seen_lines = set()
    unique_lines = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.rstrip('\n')
                if stripped_line not in seen_lines:
                    seen_lines.add(stripped_line)
                    unique_lines.append(line)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(unique_lines)
        
        print(f"Successfully removed duplicates. Output saved to '{output_file}'")
        print(f"Original lines: {len(seen_lines) + (len(unique_lines) - len(seen_lines))}")
        print(f"Unique lines: {len(unique_lines)}")
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_cleaner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)