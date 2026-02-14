
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by their extensions.
    Creates subdirectories for each file type and moves files accordingly.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()
            
            if file_extension:
                target_dir = os.path.join(directory, file_extension[1:] + "_files")
            else:
                target_dir = os.path.join(directory, "no_extension_files")
            
            os.makedirs(target_dir, exist_ok=True)
            
            try:
                shutil.move(item_path, os.path.join(target_dir, item))
                print(f"Moved: {item} -> {target_dir}")
            except Exception as e:
                print(f"Error moving {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
    print("File organization complete.")
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the specified directory by moving them into
    subfolders based on their file extensions.
    """
    # Define file type categories and their associated extensions
    file_categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c']
    }

    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Get all items in the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        # Skip if it's a directory
        if os.path.isdir(item_path):
            continue

        # Get the file extension
        file_extension = Path(item).suffix.lower()

        # Find the appropriate category for the file
        target_category = 'Others'  # Default category
        for category, extensions in file_categories.items():
            if file_extension in extensions:
                target_category = category
                break

        # Create the target category folder if it doesn't exist
        target_folder = os.path.join(directory, target_category)
        os.makedirs(target_folder, exist_ok=True)

        # Move the file to the target folder
        try:
            shutil.move(item_path, os.path.join(target_folder, item))
            print(f"Moved: {item} -> {target_category}/")
        except Exception as e:
            print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    # Use the current directory as the target, can be modified
    target_directory = os.getcwd()
    print(f"Organizing files in: {target_directory}")
    organize_files(target_directory)
    print("File organization complete.")