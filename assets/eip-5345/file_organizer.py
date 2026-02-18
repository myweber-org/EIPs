
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into folders
    named after their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
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
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the given directory by moving them into
    subdirectories named after their file extensions.
    """
    # Define categories and their associated extensions
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.md', '.xlsx', '.pptx'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv']
    }

    # Ensure the directory exists
    target_dir = Path(directory)
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Error: Directory '{directory}' does not exist or is not a directory.")
        return

    # Create category folders if they don't exist
    for category in categories:
        (target_dir / category).mkdir(exist_ok=True)

    # Track moved files and errors
    moved_files = []
    errors = []

    # Iterate over all items in the directory
    for item in target_dir.iterdir():
        # Skip directories
        if item.is_dir():
            continue

        # Get file extension
        ext = item.suffix.lower()

        # Find the appropriate category
        target_category = None
        for category, extensions in categories.items():
            if ext in extensions:
                target_category = category
                break

        # If no category found, use 'Other'
        if not target_category:
            target_category = 'Other'
            (target_dir / target_category).mkdir(exist_ok=True)

        # Construct destination path
        dest_path = target_dir / target_category / item.name

        # Move the file
        try:
            # Handle name conflicts
            if dest_path.exists():
                base_name = item.stem
                counter = 1
                while dest_path.exists():
                    new_name = f"{base_name}_{counter}{item.suffix}"
                    dest_path = target_dir / target_category / new_name
                    counter += 1

            shutil.move(str(item), str(dest_path))
            moved_files.append((item.name, target_category))
        except Exception as e:
            errors.append((item.name, str(e)))

    # Print summary
    print(f"\nOrganization complete for: {directory}")
    print(f"Files moved: {len(moved_files)}")
    print(f"Errors: {len(errors)}")

    if moved_files:
        print("\nMoved files:")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")

    if errors:
        print("\nErrors encountered:")
        for filename, error in errors:
            print(f"  {filename}: {error}")

if __name__ == "__main__":
    # Use current directory if no argument provided
    import sys
    target_directory = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    organize_files(target_directory)