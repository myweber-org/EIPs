
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the specified directory by moving them into
    subfolders named after their file extensions.
    """
    base_path = Path(directory)
    
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: The directory '{directory}' does not exist or is not a directory.")
        return

    for item in base_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            if file_extension:
                folder_name = file_extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"
            
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
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    """
    Organizes files in the specified directory by moving them into
    subfolders named after their file extensions.
    """
    base_path = Path(directory_path)

    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: The path '{directory_path}' is not a valid directory.")
        return

    for item in base_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            if file_extension:
                folder_name = file_extension[1:]  # Remove the leading dot
            else:
                folder_name = "no_extension"

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