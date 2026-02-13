
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

def organize_files(directory="."):
    """
    Organizes files in the specified directory by moving them into
    subfolders based on their file extensions.
    """
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
    
    # Track moved files and errors
    moved_files = []
    errors = []
    
    # Iterate over all items in the directory
    for item in base_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            moved = False
            
            # Find the appropriate category for the file
            for category, extensions in categories.items():
                if file_extension in extensions:
                    try:
                        destination = base_path / category / item.name
                        # Handle duplicate filenames
                        if destination.exists():
                            stem = item.stem
                            counter = 1
                            while destination.exists():
                                new_name = f"{stem}_{counter}{item.suffix}"
                                destination = base_path / category / new_name
                                counter += 1
                        
                        shutil.move(str(item), str(destination))
                        moved_files.append((item.name, category))
                        moved = True
                        break
                    except Exception as e:
                        errors.append((item.name, str(e)))
            
            # If file doesn't match any category, move to "Other"
            if not moved:
                other_dir = base_path / "Other"
                other_dir.mkdir(exist_ok=True)
                try:
                    destination = other_dir / item.name
                    if destination.exists():
                        stem = item.stem
                        counter = 1
                        while destination.exists():
                            new_name = f"{stem}_{counter}{item.suffix}"
                            destination = other_dir / new_name
                            counter += 1
                    
                    shutil.move(str(item), str(destination))
                    moved_files.append((item.name, "Other"))
                except Exception as e:
                    errors.append((item.name, str(e)))
    
    # Print summary
    if moved_files:
        print(f"Successfully organized {len(moved_files)} file(s):")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    
    if errors:
        print(f"\nEncountered {len(errors)} error(s):")
        for filename, error_msg in errors:
            print(f"  {filename}: {error_msg}")
    
    return moved_files, errors

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    organize_files()