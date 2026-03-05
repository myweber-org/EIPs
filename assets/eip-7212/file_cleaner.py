
import os
import time
from pathlib import Path

def clean_old_files(directory, days=7):
    """
    Remove files in the specified directory that are older than the given number of days.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    cutoff_time = time.time() - (days * 24 * 60 * 60)
    deleted_count = 0
    total_size = 0

    for item in Path(directory).rglob('*'):
        if item.is_file():
            try:
                file_mtime = item.stat().st_mtime
                if file_mtime < cutoff_time:
                    file_size = item.stat().st_size
                    item.unlink()
                    deleted_count += 1
                    total_size += file_size
                    print(f"Deleted: {item}")
            except (PermissionError, OSError) as e:
                print(f"Could not delete {item}: {e}")

    print(f"Deleted {deleted_count} files, freed {total_size / (1024*1024):.2f} MB.")

if __name__ == "__main__":
    target_dir = "/tmp/test_cleanup"
    clean_old_files(target_dir, days=7)