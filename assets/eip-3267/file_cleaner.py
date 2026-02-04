
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
    clean_temp_files(temp_dir)