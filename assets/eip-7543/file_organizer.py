
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organize files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return

    base_path = Path(directory_path)
    
    extension_mapping = {
        '.txt': 'TextFiles',
        '.pdf': 'Documents',
        '.jpg': 'Images',
        '.jpeg': 'Images',
        '.png': 'Images',
        '.mp3': 'Audio',
        '.mp4': 'Video',
        '.py': 'PythonScripts',
        '.zip': 'Archives'
    }

    for item in base_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            target_folder_name = extension_mapping.get(file_extension, 'Other')
            target_folder = base_path / target_folder_name
            
            target_folder.mkdir(exist_ok=True)
            
            try:
                shutil.move(str(item), str(target_folder / item.name))
                print(f"Moved: {item.name} -> {target_folder_name}/")
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files_by_extension(target_directory)