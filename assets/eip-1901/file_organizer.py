
import os
import shutil
from pathlib import Path

def organize_files(source_dir, organize_by='extension'):
    """
    Organize files in a directory by moving them into subdirectories based on their extension.
    """
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return

    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)

        if os.path.isfile(item_path):
            if organize_by == 'extension':
                file_extension = Path(item).suffix.lower()
                if file_extension:
                    target_folder = os.path.join(source_dir, file_extension[1:] + '_files')
                else:
                    target_folder = os.path.join(source_dir, 'no_extension_files')
            else:
                print(f"Organizing by '{organize_by}' is not implemented.")
                return

            os.makedirs(target_folder, exist_ok=True)
            try:
                shutil.move(item_path, os.path.join(target_folder, item))
                print(f"Moved: {item} -> {target_folder}")
            except Exception as e:
                print(f"Error moving {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    if target_directory:
        organize_files(target_directory)
    else:
        print("No directory provided.")