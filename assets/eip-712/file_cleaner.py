import os
import time
from pathlib import Path

def clean_old_files(directory, days_old):
    cutoff_time = time.time() - (days_old * 86400)
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Directory {directory} does not exist.")
        return
    
    for item in directory_path.iterdir():
        if item.is_file():
            file_stat = item.stat()
            if file_stat.st_mtime < cutoff_time:
                try:
                    item.unlink()
                    print(f"Removed: {item}")
                except Exception as e:
                    print(f"Error removing {item}: {e}")

if __name__ == "__main__":
    target_dir = "/tmp/myapp_cache"
    clean_old_files(target_dir, 7)