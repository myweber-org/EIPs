
import os
import shutil

def organize_files(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_ext = filename.split('.')[-1] if '.' in filename else 'NoExtension'
            target_dir = os.path.join(directory, file_ext.upper() + '_FILES')
            os.makedirs(target_dir, exist_ok=True)
            shutil.move(file_path, os.path.join(target_dir, filename))
            print(f"Moved: {filename} -> {target_dir}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
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
            file_ext = Path(item).suffix.lower()

            if file_ext:
                folder_name = file_ext[1:].upper() + "_Files"
            else:
                folder_name = "NO_EXTENSION_Files"

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
    Organizes files in the specified directory into subfolders based on their extensions.
    Creates folders for images, documents, archives, audio, video, and others.
    """
    # Define categories and their associated file extensions
    categories = {
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.tiff', '.webp'],
        'documents': ['.pdf', '.docx', '.doc', '.txt', '.xlsx', '.pptx', '.md', '.rtf'],
        'archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg'],
        'video': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv']
    }
    
    # Ensure the directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return
    
    # Get all files in the directory
    try:
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    except PermissionError:
        print(f"Error: Permission denied for directory '{directory_path}'.")
        return
    
    if not files:
        print("No files found to organize.")
        return
    
    moved_count = 0
    
    for filename in files:
        file_path = os.path.join(directory_path, filename)
        file_extension = Path(filename).suffix.lower()
        
        # Determine the category for the file
        target_category = 'others'
        for category, extensions in categories.items():
            if file_extension in extensions:
                target_category = category
                break
        
        # Create category folder if it doesn't exist
        category_folder = os.path.join(directory_path, target_category)
        os.makedirs(category_folder, exist_ok=True)
        
        # Move the file to the category folder
        try:
            shutil.move(file_path, os.path.join(category_folder, filename))
            moved_count += 1
            print(f"Moved: {filename} -> {target_category}/")
        except Exception as e:
            print(f"Failed to move {filename}: {e}")
    
    print(f"\nOrganization complete. Moved {moved_count} file(s).")

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    current_directory = os.getcwd()
    organize_files_by_extension(current_directory)