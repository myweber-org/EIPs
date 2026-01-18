
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organize files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory.")
        return

    base_path = Path(directory_path)
    
    extension_mapping = {
        '.txt': 'TextFiles',
        '.pdf': 'Documents',
        '.jpg': 'Images',
        '.jpeg': 'Images',
        '.png': 'Images',
        '.mp3': 'Audio',
        '.mp4': 'Video',
        '.py': 'PythonScripts',
        '.zip': 'Archives'
    }

    for item in base_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            target_folder_name = extension_mapping.get(file_extension, 'Other')
            target_folder = base_path / target_folder_name
            
            target_folder.mkdir(exist_ok=True)
            
            try:
                shutil.move(str(item), str(target_folder / item.name))
                print(f"Moved: {item.name} -> {target_folder_name}/")
            except Exception as e:
                print(f"Failed to move {item.name}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files_by_extension(target_directory)
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
        "Audio": [".mp3", ".wav", ".flac", ".aac"],
        "Video": [".mp4", ".avi", ".mov", ".mkv"],
        "Archives": [".zip", ".tar", ".gz", ".rar", ".7z"],
        "Code": [".py", ".js", ".html", ".css", ".java", ".cpp", ".c"],
    }
    
    # Create category folders if they don't exist
    for category in categories:
        (base_path / category).mkdir(exist_ok=True)
    
    # Create an 'Others' folder for uncategorized files
    others_folder = base_path / "Others"
    others_folder.mkdir(exist_ok=True)
    
    # Track moved files
    moved_files = []
    
    for item in base_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            moved = False
            
            # Find the appropriate category
            for category, extensions in categories.items():
                if file_extension in extensions:
                    target_folder = base_path / category
                    shutil.move(str(item), str(target_folder / item.name))
                    moved_files.append((item.name, category))
                    moved = True
                    break
            
            # If no category matched, move to 'Others'
            if not moved:
                shutil.move(str(item), str(others_folder / item.name))
                moved_files.append((item.name, "Others"))
    
    # Print summary
    if moved_files:
        print(f"Organized {len(moved_files)} file(s):")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    else:
        print("No files to organize.")
    
    return moved_files

if __name__ == "__main__":
    organize_files()