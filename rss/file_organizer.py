
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    """
    Organizes files in the specified directory by moving them into
    subdirectories named after their file extensions.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: The path '{directory_path}' is not a valid directory.")
        return

    base_path = Path(directory_path)

    for item in base_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            if file_extension:
                # Remove the leading dot from the extension for folder naming
                folder_name = file_extension[1:] if file_extension.startswith('.') else file_extension
                target_folder = base_path / folder_name
                target_folder.mkdir(exist_ok=True)
                try:
                    shutil.move(str(item), str(target_folder / item.name))
                    print(f"Moved: {item.name} -> {folder_name}/")
                except Exception as e:
                    print(f"Failed to move {item.name}: {e}")
            else:
                print(f"Skipped '{item.name}' (no extension).")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)