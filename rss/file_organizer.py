
import os
import shutil
from pathlib import Path

def organize_files(directory="."):
    """
    Organizes files in the specified directory by moving them into
    subfolders named after their file extensions.
    """
    base_path = Path(directory).resolve()

    # Define categories and their associated extensions
    categories = {
        "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg"],
        "Documents": [".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md"],
        "Archives": [".zip", ".tar", ".gz", ".7z", ".rar"],
        "Code": [".py", ".js", ".html", ".css", ".json", ".xml"],
        "Audio": [".mp3", ".wav", ".flac", ".aac"],
        "Video": [".mp4", ".avi", ".mov", ".mkv"]
    }

    # Create a reverse lookup: extension -> category
    ext_to_category = {}
    for category, extensions in categories.items():
        for ext in extensions:
            ext_to_category[ext] = category

    # Iterate over all items in the directory
    for item in base_path.iterdir():
        if item.is_file():
            file_ext = item.suffix.lower()
            target_category = ext_to_category.get(file_ext, "Other")

            # Create target directory if it doesn't exist
            target_dir = base_path / target_category
            target_dir.mkdir(exist_ok=True)

            # Move the file
            try:
                shutil.move(str(item), str(target_dir / item.name))
                print(f"Moved: {item.name} -> {target_category}/")
            except Exception as e:
                print(f"Error moving {item.name}: {e}")

if __name__ == "__main__":
    organize_files()