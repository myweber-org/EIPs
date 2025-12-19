
import os
import sys
from datetime import datetime

def rename_files_by_date(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return False
    
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    for filename in files:
        filepath = os.path.join(directory, filename)
        stat = os.stat(filepath)
        creation_time = datetime.fromtimestamp(stat.st_ctime)
        new_name = creation_time.strftime("%Y%m%d_%H%M%S") + os.path.splitext(filename)[1]
        new_path = os.path.join(directory, new_name)
        
        counter = 1
        while os.path.exists(new_path):
            name_part, ext_part = os.path.splitext(new_name)
            new_name = f"{name_part}_{counter}{ext_part}"
            new_path = os.path.join(directory, new_name)
            counter += 1
        
        try:
            os.rename(filepath, new_path)
            print(f"Renamed: {filename} -> {new_name}")
        except Exception as e:
            print(f"Failed to rename {filename}: {e}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python file_renamer.py <directory_path>")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    rename_files_by_date(target_dir)