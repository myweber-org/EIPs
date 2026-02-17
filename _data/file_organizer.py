
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into
    subfolders named after their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        return

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        if os.path.isfile(item_path):
            file_ext = Path(item).suffix.lower()

            if file_ext:
                folder_name = file_ext[1:] if file_ext.startswith('.') else file_ext
            else:
                folder_name = "no_extension"

            target_folder = os.path.join(directory, folder_name)
            os.makedirs(target_folder, exist_ok=True)

            try:
                shutil.move(item_path, os.path.join(target_folder, item))
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_dir = input("Enter the directory path to organize: ").strip()
    organize_files(target_dir)
    print("File organization complete.")
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organizes files in the specified directory by moving them into
    subfolders named after their file extensions.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: The path '{directory_path}' is not a valid directory.")
        return

    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)

        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()

            if not file_extension:
                folder_name = "no_extension"
            else:
                folder_name = file_extension[1:]

            target_folder = os.path.join(directory_path, folder_name)
            os.makedirs(target_folder, exist_ok=True)

            try:
                shutil.move(item_path, os.path.join(target_folder, item))
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files_by_extension(target_directory)