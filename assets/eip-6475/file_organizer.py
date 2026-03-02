
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organize files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    # Define file type categories
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.tar', '.gz', '.rar'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
    }
    
    # Ensure the directory exists
    target_dir = Path(directory)
    if not target_dir.exists():
        print(f"Directory {directory} does not exist.")
        return
    
    # Create category folders if they don't exist
    for category in categories:
        category_path = target_dir / category
        category_path.mkdir(exist_ok=True)
    
    # Track moved files
    moved_files = []
    skipped_files = []
    
    # Iterate through files in the directory
    for item in target_dir.iterdir():
        if item.is_file():
            file_ext = item.suffix.lower()
            moved = False
            
            # Find the appropriate category
            for category, extensions in categories.items():
                if file_ext in extensions:
                    dest_dir = target_dir / category
                    try:
                        shutil.move(str(item), str(dest_dir / item.name))
                        moved_files.append((item.name, category))
                        moved = True
                        break
                    except Exception as e:
                        print(f"Error moving {item.name}: {e}")
                        skipped_files.append(item.name)
                        break
            
            # If file doesn't match any category, move to 'Other'
            if not moved:
                other_dir = target_dir / 'Other'
                other_dir.mkdir(exist_ok=True)
                try:
                    shutil.move(str(item), str(other_dir / item.name))
                    moved_files.append((item.name, 'Other'))
                except Exception as e:
                    print(f"Error moving {item.name}: {e}")
                    skipped_files.append(item.name)
    
    # Print summary
    print(f"\nOrganization complete!")
    print(f"Moved {len(moved_files)} files:")
    for filename, category in moved_files:
        print(f"  {filename} -> {category}/")
    
    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} files:")
        for filename in skipped_files:
            print(f"  {filename}")

if __name__ == "__main__":
    # Use current directory if no argument provided
    import sys
    target_directory = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
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
    
    extensions_folders = {
        '.txt': 'TextFiles',
        '.pdf': 'PDFs',
        '.jpg': 'Images',
        '.png': 'Images',
        '.mp3': 'Audio',
        '.mp4': 'Videos',
        '.py': 'PythonScripts',
        '.zip': 'Archives'
    }
    
    for item in source_path.iterdir():
        if item.is_file():
            file_ext = item.suffix.lower()
            folder_name = extensions_folders.get(file_ext, 'OtherFiles')
            
            dest_folder = target_path / folder_name
            dest_folder.mkdir(exist_ok=True)
            
            try:
                shutil.move(str(item), str(dest_folder / item.name))
                print(f"Moved {item.name} to {folder_name}/")
            except Exception as e:
                print(f"Error moving {item.name}: {e}")
    
    print("File organization completed.")

if __name__ == "__main__":
    organize_files("./downloads")