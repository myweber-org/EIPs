
import os
import glob
from pathlib import Path

def rename_files_sequentially(directory, prefix="file", extension=".txt"):
    files = glob.glob(os.path.join(directory, "*" + extension))
    files.sort(key=os.path.getctime)
    
    for index, file_path in enumerate(files, start=1):
        new_name = f"{prefix}_{index:03d}{extension}"
        new_path = os.path.join(directory, new_name)
        os.rename(file_path, new_path)
        print(f"Renamed: {Path(file_path).name} -> {new_name}")

if __name__ == "__main__":
    target_dir = input("Enter directory path: ").strip()
    if os.path.isdir(target_dir):
        rename_files_sequentially(target_dir)
    else:
        print("Invalid directory path")