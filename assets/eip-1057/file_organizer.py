
import os
import shutil

def organize_files(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            file_extension = filename.split('.')[-1] if '.' in filename else 'no_extension'
            target_dir = os.path.join(directory, file_extension)
            
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            shutil.move(file_path, os.path.join(target_dir, filename))
            print(f"Moved {filename} to {file_extension}/")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil

def organize_files(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            file_extension = filename.split('.')[-1] if '.' in filename else 'no_extension'
            target_folder = os.path.join(directory, file_extension)
            
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            
            shutil.move(file_path, os.path.join(target_folder, filename))
            print(f"Moved {filename} to {file_extension}/")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ")
    
    if os.path.exists(target_directory):
        organize_files(target_directory)
        print("File organization completed.")
    else:
        print("Directory does not exist.")
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the given directory by moving them into
    subfolders based on their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        return

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()

            if not file_extension:
                folder_name = "NoExtension"
            else:
                folder_name = file_extension[1:].capitalize() + "Files"

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