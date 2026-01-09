
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