
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
        'Archives': ['.zip', '.rar', '.tar', '.gz'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp'],
    }

    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Create category folders if they don't exist
    for category in file_categories.keys():
        category_path = os.path.join(directory, category)
        os.makedirs(category_path, exist_ok=True)

    # Track moved files and errors
    moved_files = []
    error_files = []

    # Iterate over all items in the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        # Skip directories
        if os.path.isdir(item_path):
            continue

        # Get file extension
        file_extension = Path(item).suffix.lower()

        # Find the appropriate category
        target_category = None
        for category, extensions in file_categories.items():
            if file_extension in extensions:
                target_category = category
                break

        # If no category found, use 'Other'
        if not target_category:
            target_category = 'Other'
            other_folder = os.path.join(directory, target_category)
            os.makedirs(other_folder, exist_ok=True)

        # Move the file
        try:
            target_path = os.path.join(directory, target_category, item)
            shutil.move(item_path, target_path)
            moved_files.append((item, target_category))
        except Exception as e:
            error_files.append((item, str(e)))

    # Print summary
    print(f"Organization complete for: {directory}")
    if moved_files:
        print(f"\nMoved {len(moved_files)} files:")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    if error_files:
        print(f"\nEncountered {len(error_files)} errors:")
        for filename, error in error_files:
            print(f"  {filename}: {error}")

if __name__ == "__main__":
    target_dir = input("Enter the directory path to organize: ").strip()
    organize_files(target_dir)