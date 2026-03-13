
import os
import shutil

def organize_files(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            _, extension = os.path.splitext(filename)
            extension = extension.lower()
            
            if extension:
                folder_name = extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"
            
            target_folder = os.path.join(directory, folder_name)
            os.makedirs(target_folder, exist_ok=True)
            
            target_path = os.path.join(target_folder, filename)
            shutil.move(file_path, target_path)
            print(f"Moved: {filename} -> {folder_name}/")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organize files in the given directory into subfolders based on their extensions.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: The path '{directory_path}' is not a valid directory.")
        return

    path = Path(directory_path)
    
    # Define categories and their associated file extensions
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.tiff'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md', '.rtf'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg'],
        'Video': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv'],
        'Archives': ['.zip', '.rar', '.7z', '.tar', '.gz'],
        'Scripts': ['.py', '.js', '.html', '.css', '.php', '.sh', '.bat'],
        'Executables': ['.exe', '.msi', '.app', '.dmg'],
        'Data': ['.csv', '.json', '.xml', '.sql', '.db']
    }
    
    # Create a mapping from extension to category
    extension_to_category = {}
    for category, extensions in categories.items():
        for ext in extensions:
            extension_to_category[ext.lower()] = category
    
    # Process each file in the directory
    for item in path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            category = extension_to_category.get(file_extension, 'Other')
            
            # Create category folder if it doesn't exist
            category_folder = path / category
            category_folder.mkdir(exist_ok=True)
            
            # Move the file to the category folder
            try:
                destination = category_folder / item.name
                if not destination.exists():
                    shutil.move(str(item), str(category_folder))
                    print(f"Moved: {item.name} -> {category}/")
                else:
                    print(f"Skipped: {item.name} already exists in {category}/")
            except Exception as e:
                print(f"Error moving {item.name}: {e}")
    
    print("File organization completed.")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files_by_extension(target_directory)
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organize files in the given directory by moving them into folders
    named after their file extensions.
    """
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()
            
            if file_extension:
                folder_name = file_extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"
            
            target_folder = os.path.join(directory_path, folder_name)
            os.makedirs(target_folder, exist_ok=True)
            
            try:
                shutil.move(item_path, os.path.join(target_folder, item))
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Error moving {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files_by_extension(target_directory)
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