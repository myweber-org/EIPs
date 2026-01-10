
import os
import glob
from pathlib import Path
from datetime import datetime

def rename_files_sequentially(directory, prefix="file", extension=".txt"):
    files = glob.glob(os.path.join(directory, "*" + extension))
    files.sort(key=os.path.getctime)
    
    for index, filepath in enumerate(files, start=1):
        new_filename = f"{prefix}_{index:03d}{extension}"
        new_filepath = os.path.join(directory, new_filename)
        os.rename(filepath, new_filepath)
        print(f"Renamed: {Path(filepath).name} -> {new_filename}")

if __name__ == "__main__":
    target_dir = "./documents"
    if os.path.exists(target_dir):
        rename_files_sequentially(target_dir, prefix="document", extension=".pdf")
    else:
        print(f"Directory {target_dir} does not exist.")