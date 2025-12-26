
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by their extensions.
    Creates subdirectories for each file type and moves files accordingly.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()
            
            if file_extension:
                target_dir = os.path.join(directory, file_extension[1:])
            else:
                target_dir = os.path.join(directory, "no_extension")
            
            os.makedirs(target_dir, exist_ok=True)
            
            try:
                shutil.move(item_path, os.path.join(target_dir, item))
                print(f"Moved: {item} -> {target_dir}")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
    print("File organization completed.")