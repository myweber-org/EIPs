
import os
import sys
from datetime import datetime

def rename_files_with_timestamp(directory_path):
    """
    Rename all files in the given directory by adding a timestamp prefix.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return False
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    renamed_count = 0
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        if os.path.isfile(file_path):
            name, ext = os.path.splitext(filename)
            new_filename = f"{timestamp}_{name}{ext}"
            new_file_path = os.path.join(directory_path, new_filename)
            
            try:
                os.rename(file_path, new_file_path)
                print(f"Renamed: {filename} -> {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"Failed to rename {filename}: {e}")
    
    print(f"Total files renamed: {renamed_count}")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python file_renamer.py <directory_path>")
        sys.exit(1)
    
    target_directory = sys.argv[1]
    rename_files_with_timestamp(target_directory)