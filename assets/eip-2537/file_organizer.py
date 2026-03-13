
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
        print(f"Error: Directory '{directory}' does not exist or is not a directory.")
        return

    for item in base_path.iterdir():
        if item.is_file():
            extension = item.suffix.lower()
            if extension:
                folder_name = extension[1:] if extension.startswith('.') else extension
                target_folder = base_path / folder_name
                target_folder.mkdir(exist_ok=True)
                try:
                    shutil.move(str(item), str(target_folder / item.name))
                    print(f"Moved: {item.name} -> {folder_name}/")
                except Exception as e:
                    print(f"Failed to move {item.name}: {e}")
            else:
                print(f"Skipped {item.name} (no extension)")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)