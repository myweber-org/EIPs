
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organizes files in the specified directory by moving them into
    subfolders named after their file extensions.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: The path '{directory_path}' is not a valid directory.")
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
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files_by_extension(target_directory)
import os
import shutil

def organize_files(directory):
    """
    Organize files in the given directory into subdirectories based on their extensions.
    """
    # Define categories and their associated file extensions
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Audio': ['.mp3', '.wav', '.aac', '.flac'],
        'Video': ['.mp4', '.mkv', '.avi', '.mov', '.wmv'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c'],
        'Executables': ['.exe', '.msi', '.sh', '.bat']
    }

    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Get all files in the directory
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    except PermissionError:
        print(f"Error: Permission denied for directory '{directory}'.")
        return

    moved_count = 0

    for filename in files:
        file_path = os.path.join(directory, filename)
        # Get the file extension
        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        # Find the category for the extension
        target_category = None
        for category, extensions in categories.items():
            if ext in extensions:
                target_category = category
                break

        # If no category found, put in 'Other'
        if target_category is None:
            target_category = 'Other'

        # Create target directory if it doesn't exist
        target_dir = os.path.join(directory, target_category)
        if not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir)
            except OSError as e:
                print(f"Error creating directory '{target_dir}': {e}")
                continue

        # Move the file
        try:
            shutil.move(file_path, os.path.join(target_dir, filename))
            moved_count += 1
            print(f"Moved: {filename} -> {target_category}/")
        except shutil.Error as e:
            print(f"Error moving file '{filename}': {e}")
        except PermissionError:
            print(f"Permission denied moving file '{filename}'.")

    print(f"\nOrganization complete. Moved {moved_count} file(s).")

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    target_directory = input("Enter the directory path to organize (or press Enter for current directory): ").strip()
    if not target_directory:
        target_directory = os.getcwd()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(directory_path):
    """
    Organizes files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    # Define file type categories and their associated extensions
    file_categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md', '.rtf'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg'],
        'Video': ['.mp4', '.avi', '.mkv', '.mov', '.wmv'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.json'],
        'Executables': ['.exe', '.msi', '.bat', '.sh'],
    }

    # Ensure the directory path exists
    path = Path(directory_path)
    if not path.exists() or not path.is_dir():
        print(f"Error: The directory '{directory_path}' does not exist.")
        return

    # Create category folders if they don't exist
    for category in file_categories.keys():
        category_folder = path / category
        category_folder.mkdir(exist_ok=True)

    # Create an 'Other' folder for uncategorized files
    other_folder = path / 'Other'
    other_folder.mkdir(exist_ok=True)

    # Track moved files count
    moved_files_count = 0

    # Iterate over all items in the directory
    for item in path.iterdir():
        # Skip directories
        if item.is_dir():
            continue

        # Get file extension
        file_extension = item.suffix.lower()

        # Determine the target category
        target_category = None
        for category, extensions in file_categories.items():
            if file_extension in extensions:
                target_category = category
                break

        # If no category matched, use 'Other'
        if target_category is None:
            target_category = 'Other'

        # Construct target path
        target_folder = path / target_category
        target_path = target_folder / item.name

        # Avoid overwriting existing files
        if target_path.exists():
            base_name = item.stem
            counter = 1
            while target_path.exists():
                new_name = f"{base_name}_{counter}{item.suffix}"
                target_path = target_folder / new_name
                counter += 1

        # Move the file
        try:
            shutil.move(str(item), str(target_path))
            moved_files_count += 1
            print(f"Moved: {item.name} -> {target_category}/")
        except Exception as e:
            print(f"Failed to move {item.name}: {e}")

    print(f"\nOrganization complete. Moved {moved_files_count} file(s).")

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    current_directory = os.getcwd()
    organize_files(current_directory)
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
    
    for item in path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            
            if file_extension:
                folder_name = file_extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"
            
            target_folder = path / folder_name
            target_folder.mkdir(exist_ok=True)
            
            try:
                shutil.move(str(item), str(target_folder / item.name))
                print(f"Moved: {item.name} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files_by_extension(target_directory)