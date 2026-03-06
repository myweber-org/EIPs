
import os
import shutil
from pathlib import Path

def organize_files(directory="."):
    """Organize files in the given directory by their extensions."""
    base_path = Path(directory).resolve()
    
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: Directory '{directory}' does not exist or is not a directory.")
        return

    extension_categories = {
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'documents': ['.pdf', '.docx', '.txt', '.md', '.xlsx', '.pptx'],
        'audio': ['.mp3', '.wav', '.flac', '.aac'],
        'video': ['.mp4', '.avi', '.mkv', '.mov'],
        'archives': ['.zip', '.tar', '.gz', '.7z'],
        'code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
    }

    other_folder = base_path / 'other'
    other_folder.mkdir(exist_ok=True)

    for item in base_path.iterdir():
        if item.is_file():
            file_ext = item.suffix.lower()
            moved = False

            for category, extensions in extension_categories.items():
                if file_ext in extensions:
                    category_folder = base_path / category
                    category_folder.mkdir(exist_ok=True)
                    try:
                        shutil.move(str(item), str(category_folder / item.name))
                        print(f"Moved: {item.name} -> {category}/")
                        moved = True
                        break
                    except Exception as e:
                        print(f"Failed to move {item.name}: {e}")

            if not moved:
                try:
                    shutil.move(str(item), str(other_folder / item.name))
                    print(f"Moved: {item.name} -> other/")
                except Exception as e:
                    print(f"Failed to move {item.name}: {e}")

    print("File organization completed.")

if __name__ == "__main__":
    target_dir = input("Enter directory path to organize (press Enter for current): ").strip()
    if not target_dir:
        target_dir = "."
    organize_files(target_dir)
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into subdirectories
    based on their file extensions.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()
            
            if file_extension:
                target_dir = os.path.join(directory, file_extension[1:])
            else:
                target_dir = os.path.join(directory, "no_extension")
            
            os.makedirs(target_dir, exist_ok=True)
            
            try:
                shutil.move(item_path, os.path.join(target_dir, item))
                print(f"Moved: {item} -> {target_dir}")
            except Exception as e:
                print(f"Error moving {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)
    print("File organization completed.")