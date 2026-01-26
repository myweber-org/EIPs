
import os
import sys
from datetime import datetime

def rename_files_by_date(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            try:
                stat = os.stat(filepath)
                creation_time = stat.st_ctime
                date_str = datetime.fromtimestamp(creation_time).strftime("%Y%m%d_%H%M%S")
                name, ext = os.path.splitext(filename)
                new_filename = f"{date_str}{ext}"
                new_filepath = os.path.join(directory, new_filename)
                os.rename(filepath, new_filepath)
                print(f"Renamed: {filename} -> {new_filename}")
            except Exception as e:
                print(f"Failed to rename {filename}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python file_renamer.py <directory_path>")
        sys.exit(1)
    target_directory = sys.argv[1]
    rename_files_by_date(target_directory)