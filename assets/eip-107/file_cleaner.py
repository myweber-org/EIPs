import os
import re
import sys

def normalize_filename(filename):
    """Convert filename to lowercase and replace spaces with underscores."""
    name, ext = os.path.splitext(filename)
    normalized = re.sub(r'\s+', '_', name.lower())
    return f"{normalized}{ext}"

def clean_directory(directory_path):
    """Rename all files in the directory with normalized names."""
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return

    for filename in os.listdir(directory_path):
        old_path = os.path.join(directory_path, filename)
        if os.path.isfile(old_path):
            new_name = normalize_filename(filename)
            new_path = os.path.join(directory_path, new_name)
            if old_path != new_path:
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} -> {new_name}")
                except OSError as e:
                    print(f"Failed to rename {filename}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python file_cleaner.py <directory_path>")
        sys.exit(1)
    target_dir = sys.argv[1]
    clean_directory(target_dir)