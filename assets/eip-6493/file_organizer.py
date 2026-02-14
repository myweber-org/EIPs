
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organize files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory.")
        return

    path = Path(directory_path)
    
    for item in path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            
            if file_extension:
                folder_name = file_extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"
            
            target_folder = path / folder_name
            target_folder.mkdir(exist_ok=True)
            
            try:
                shutil.move(str(item), str(target_folder / item.name))
                print(f"Moved: {item.name} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files_by_extension(target_directory)
import os
import shutil

def organize_files(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            file_extension = filename.split('.')[-1] if '.' in filename else 'no_extension'
            target_dir = os.path.join(directory, file_extension)
            
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            shutil.move(file_path, os.path.join(target_dir, filename))
            print(f"Moved {filename} to {file_extension}/")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)