import os
import time
import shutil

def clean_old_files(directory, days=7):
    """
    Remove files in the given directory that are older than the specified number of days.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    current_time = time.time()
    cutoff_time = current_time - (days * 24 * 60 * 60)
    removed_count = 0
    error_count = 0

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                file_mtime = os.path.getmtime(file_path)
                if file_mtime < cutoff_time:
                    os.remove(file_path)
                    removed_count += 1
                    print(f"Removed: {file_path}")
            elif os.path.isdir(file_path):
                dir_mtime = os.path.getmtime(file_path)
                if dir_mtime < cutoff_time:
                    shutil.rmtree(file_path)
                    removed_count += 1
                    print(f"Removed directory: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            error_count += 1

    print(f"Cleanup completed. Removed {removed_count} items. Encountered {error_count} errors.")

if __name__ == "__main__":
    target_dir = "/tmp/my_temp_files"
    clean_old_files(target_dir, days=7)