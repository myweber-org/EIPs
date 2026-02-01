
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    """
    Organizes files in the given directory by moving them into
    subdirectories named after their file extensions.
    """
    path = Path(directory_path)

    if not path.exists() or not path.is_dir():
        print(f"Error: The path '{directory_path}' is not a valid directory.")
        return

    for item in path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            if file_extension:
                target_dir = path / file_extension[1:]
            else:
                target_dir = path / "no_extension"

            target_dir.mkdir(exist_ok=True)
            try:
                shutil.move(str(item), str(target_dir / item.name))
                print(f"Moved: {item.name} -> {target_dir.name}/")
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)