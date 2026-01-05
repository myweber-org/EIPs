
import os
import shutil
import sys

def clean_temp_files(directory, extensions=('.tmp', '.temp', '.log', '.cache')):
    """
    Remove temporary files with specified extensions from a directory.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return False

    removed_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extensions):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                    removed_count += 1
                except OSError as e:
                    print(f"Error removing {file_path}: {e}")

    print(f"Cleaning complete. Removed {removed_count} files.")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_cleaner.py <directory_path>")
        sys.exit(1)

    target_dir = sys.argv[1]
    clean_temp_files(target_dir)