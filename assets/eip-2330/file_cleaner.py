
import os
import shutil
import tempfile
from pathlib import Path

def clean_temp_files(directory: str, extensions: tuple = ('.tmp', '.temp', '.log'), days_old: int = 7):
    """
    Remove temporary files with specified extensions older than given days.
    """
    target_dir = Path(directory)
    if not target_dir.exists() or not target_dir.is_dir():
        raise ValueError(f"Invalid directory: {directory}")

    current_time = time.time()
    cutoff_time = current_time - (days_old * 24 * 60 * 60)

    removed_count = 0
    total_size = 0

    for file_path in target_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            file_stat = file_path.stat()
            if file_stat.st_mtime < cutoff_time:
                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    removed_count += 1
                    total_size += file_size
                    print(f"Removed: {file_path.name} ({file_size} bytes)")
                except OSError as e:
                    print(f"Failed to remove {file_path.name}: {e}")

    print(f"Cleanup completed: {removed_count} files removed, {total_size} bytes freed")

if __name__ == "__main__":
    import time
    temp_dir = tempfile.gettempdir()
    print(f"Cleaning temporary directory: {temp_dir}")
    clean_temp_files(temp_dir)