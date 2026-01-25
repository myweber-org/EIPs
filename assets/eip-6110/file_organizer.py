
import os
import shutil
from pathlib import Path

def organize_files(directory="."):
    """Organize files in the given directory by their extensions."""
    base_path = Path(directory)
    
    # Define categories and their associated file extensions
    categories = {
        "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg"],
        "Documents": [".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md"],
        "Archives": [".zip", ".tar", ".gz", ".rar", ".7z"],
        "Code": [".py", ".js", ".html", ".css", ".java", ".cpp", ".c"],
        "Audio": [".mp3", ".wav", ".flac", ".aac"],
        "Video": [".mp4", ".avi", ".mov", ".mkv"],
    }
    
    # Create category folders if they don't exist
    for category in categories:
        (base_path / category).mkdir(exist_ok=True)
    
    # Create an 'Others' folder for uncategorized files
    others_folder = base_path / "Others"
    others_folder.mkdir(exist_ok=True)
    
    # Organize files
    for item in base_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            moved = False
            
            # Check each category for a matching extension
            for category, extensions in categories.items():
                if file_extension in extensions:
                    target_folder = base_path / category
                    shutil.move(str(item), str(target_folder / item.name))
                    moved = True
                    break
            
            # If no category matched, move to 'Others'
            if not moved:
                shutil.move(str(item), str(others_folder / item.name))
    
    print(f"Files in '{directory}' have been organized.")

if __name__ == "__main__":
    # Organize files in the current directory
    organize_files()