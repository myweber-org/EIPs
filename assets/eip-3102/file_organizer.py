
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into folders
    named after their file extensions.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()
            
            if file_extension:
                folder_name = file_extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"
            
            folder_path = os.path.join(directory, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            destination_path = os.path.join(folder_path, item)
            shutil.move(item_path, destination_path)
            print(f"Moved: {item} -> {folder_name}/")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    """
    Organizes files in the given directory by moving them into
    subfolders named after their file extensions.
    """
    base_path = Path(directory_path)
    
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: The path '{directory_path}' is not a valid directory.")
        return

    for item in base_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            if not file_extension:
                folder_name = "no_extension"
            else:
                folder_name = file_extension[1:] + "_files"

            target_folder = base_path / folder_name
            target_folder.mkdir(exist_ok=True)

            try:
                shutil.move(str(item), str(target_folder / item.name))
                print(f"Moved: {item.name} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)