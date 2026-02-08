
import os
import glob
from pathlib import Path
from datetime import datetime

def rename_files_in_directory(directory_path, prefix="file"):
    try:
        files = list(Path(directory_path).glob("*"))
        files_with_time = []
        
        for file_path in files:
            if file_path.is_file():
                creation_time = file_path.stat().st_ctime
                files_with_time.append((creation_time, file_path))
        
        files_with_time.sort(key=lambda x: x[0])
        
        for index, (_, file_path) in enumerate(files_with_time, start=1):
            extension = file_path.suffix
            new_name = f"{prefix}_{index:03d}{extension}"
            new_path = file_path.parent / new_name
            
            if new_path.exists():
                print(f"Warning: {new_name} already exists. Skipping rename.")
                continue
                
            file_path.rename(new_path)
            print(f"Renamed: {file_path.name} -> {new_name}")
            
        print(f"Successfully renamed {len(files_with_time)} files.")
        
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path: ").strip()
    
    if os.path.isdir(target_directory):
        custom_prefix = input("Enter file prefix (default: 'file'): ").strip()
        prefix = custom_prefix if custom_prefix else "file"
        rename_files_in_directory(target_directory, prefix)
    else:
        print("Invalid directory path.")