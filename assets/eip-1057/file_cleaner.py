import os
import shutil
import sys

def clean_temp_files(directory, extensions=('.tmp', '.temp', '.log', '.cache')):
    """
    Remove temporary files with specified extensions from a directory.
    """
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    if not os.path.isdir(directory):
        print(f"{directory} is not a directory.")
        return

    removed_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extensions):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    removed_files.append(file_path)
                    print(f"Removed: {file_path}")
                except OSError as e:
                    print(f"Error removing {file_path}: {e}")

    print(f"Total files removed: {len(removed_files)}")
    return removed_files

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_cleaner.py <directory> [extensions]")
        sys.exit(1)

    target_dir = sys.argv[1]
    exts = tuple(sys.argv[2].split(',')) if len(sys.argv) > 2 else ('.tmp', '.temp', '.log', '.cache')
    clean_temp_files(target_dir, exts)