
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
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
                shutil.move(item_path, target_folder)
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from pathlib import Path

def organize_files(source_dir, target_dir=None):
    if target_dir is None:
        target_dir = source_dir
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"Source directory {source_dir} does not exist.")
        return
    
    target_path.mkdir(parents=True, exist_ok=True)
    
    file_categories = {
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'audio': ['.mp3', '.wav', '.flac', '.aac'],
        'video': ['.mp4', '.avi', '.mov', '.mkv'],
        'archives': ['.zip', '.tar', '.gz', '.rar'],
        'code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
    }
    
    for file_path in source_path.iterdir():
        if file_path.is_file():
            file_extension = file_path.suffix.lower()
            category_found = False
            
            for category, extensions in file_categories.items():
                if file_extension in extensions:
                    category_dir = target_path / category
                    category_dir.mkdir(exist_ok=True)
                    
                    try:
                        shutil.move(str(file_path), str(category_dir / file_path.name))
                        print(f"Moved {file_path.name} to {category}/")
                    except Exception as e:
                        print(f"Error moving {file_path.name}: {e}")
                    
                    category_found = True
                    break
            
            if not category_found:
                other_dir = target_path / 'other'
                other_dir.mkdir(exist_ok=True)
                
                try:
                    shutil.move(str(file_path), str(other_dir / file_path.name))
                    print(f"Moved {file_path.name} to other/")
                except Exception as e:
                    print(f"Error moving {file_path.name}: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        source_directory = sys.argv[1]
        target_directory = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        source_directory = input("Enter source directory path: ").strip()
        target_input = input("Enter target directory path (press Enter to use source): ").strip()
        target_directory = target_input if target_input else None
    
    organize_files(source_directory, target_directory)
    print("File organization completed.")