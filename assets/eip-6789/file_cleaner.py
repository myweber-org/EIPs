import os
import time
from pathlib import Path

def clean_old_files(directory, days=7):
    """
    Remove files older than specified days from given directory.
    """
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    
    cutoff_time = time.time() - (days * 24 * 60 * 60)
    removed_count = 0
    
    for item in Path(directory).iterdir():
        if item.is_file():
            file_mtime = item.stat().st_mtime
            if file_mtime < cutoff_time:
                try:
                    item.unlink()
                    removed_count += 1
                    print(f"Removed: {item.name}")
                except OSError as e:
                    print(f"Error removing {item.name}: {e}")
    
    print(f"Cleanup complete. Removed {removed_count} file(s).")

if __name__ == "__main__":
    target_dir = "/tmp/test_files"
    clean_old_files(target_dir, days=7)
import sys
import os

def remove_duplicate_lines(input_file, output_file=None):
    """
    Remove duplicate lines from a text file while preserving order.
    If output_file is not specified, overwrite the input file.
    """
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return False
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        seen = set()
        unique_lines = []
        
        for line in lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        
        if output_file is None:
            output_file = input_file
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(unique_lines)
        
        removed_count = len(lines) - len(unique_lines)
        print(f"Removed {removed_count} duplicate line(s).")
        print(f"Original: {len(lines)} lines, Unique: {len(unique_lines)} lines.")
        
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
    
    success = remove_duplicate_lines(input_file, output_file)
    sys.exit(0 if success else 1)