
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    """
    Organize files in the given directory by their extensions.
    Creates subdirectories for each file type and moves files accordingly.
    """
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    path = Path(directory_path)
    
    for item in path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            if file_extension:
                target_dir = path / file_extension[1:]
            else:
                target_dir = path / "no_extension"
            
            target_dir.mkdir(exist_ok=True)
            shutil.move(str(item), str(target_dir / item.name))
    
    print(f"Files in '{directory_path}' have been organized.")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the specified directory by moving them into
    subfolders named after their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()

            if file_extension:
                folder_name = file_extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"

            target_folder = os.path.join(directory, folder_name)
            os.makedirs(target_folder, exist_ok=True)

            try:
                shutil.move(item_path, os.path.join(target_folder, item))
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
    print("File organization complete.")