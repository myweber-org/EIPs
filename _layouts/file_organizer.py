
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into
    subfolders based on their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()

            if not file_extension:
                folder_name = "NoExtension"
            else:
                folder_name = file_extension[1:].capitalize()

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
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organize files in the specified directory by moving them into
    subfolders based on their file extensions.
    """
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return
    
    path = Path(directory_path)
    
    extension_folders = {
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'audio': ['.mp3', '.wav', '.flac', '.aac'],
        'video': ['.mp4', '.avi', '.mov', '.mkv', '.flv'],
        'archives': ['.zip', '.rar', '.tar', '.gz', '.7z'],
        'code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c']
    }
    
    for item in path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            destination_folder = None
            
            for folder, extensions in extension_folders.items():
                if file_extension in extensions:
                    destination_folder = folder
                    break
            
            if not destination_folder:
                destination_folder = 'others'
            
            destination_path = path / destination_folder
            destination_path.mkdir(exist_ok=True)
            
            try:
                shutil.move(str(item), str(destination_path / item.name))
                print(f"Moved: {item.name} -> {destination_folder}/")
            except Exception as e:
                print(f"Error moving {item.name}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files_by_extension(target_directory)
import os
import shutil

def organize_files(directory):
    """
    Organize files in the specified directory by moving them into
    subdirectories named after their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            file_ext = os.path.splitext(filename)[1].lower()

            if file_ext:
                target_dir = os.path.join(directory, file_ext[1:])
            else:
                target_dir = os.path.join(directory, "no_extension")

            os.makedirs(target_dir, exist_ok=True)
            shutil.move(file_path, os.path.join(target_dir, filename))
            print(f"Moved: {filename} -> {target_dir}/")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)