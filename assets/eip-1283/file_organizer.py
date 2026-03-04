
import os
import shutil
from pathlib import Path

def organize_files(source_dir, target_base_dir=None):
    """
    Organize files in source directory by their extensions.
    Creates subdirectories for each file type.
    """
    if target_base_dir is None:
        target_base_dir = source_dir
    
    source_path = Path(source_dir)
    target_path = Path(target_base_dir)
    
    if not source_path.exists():
        print(f"Source directory {source_dir} does not exist.")
        return
    
    if not target_path.exists():
        target_path.mkdir(parents=True)
    
    extension_categories = {
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'documents': ['.pdf', '.docx', '.txt', '.md', '.rtf', '.odt'],
        'spreadsheets': ['.xlsx', '.xls', '.csv', '.ods'],
        'presentations': ['.pptx', '.ppt', '.odp'],
        'archives': ['.zip', '.tar', '.gz', '.7z', '.rar'],
        'code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c'],
        'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg'],
        'video': ['.mp4', '.avi', '.mkv', '.mov', '.wmv']
    }
    
    other_dir = target_path / 'other'
    other_dir.mkdir(exist_ok=True)
    
    for file_path in source_path.iterdir():
        if file_path.is_file():
            file_extension = file_path.suffix.lower()
            moved = False
            
            for category, extensions in extension_categories.items():
                if file_extension in extensions:
                    category_dir = target_path / category
                    category_dir.mkdir(exist_ok=True)
                    
                    try:
                        shutil.move(str(file_path), str(category_dir / file_path.name))
                        print(f"Moved {file_path.name} to {category}/")
                        moved = True
                        break
                    except Exception as e:
                        print(f"Error moving {file_path.name}: {e}")
            
            if not moved:
                try:
                    shutil.move(str(file_path), str(other_dir / file_path.name))
                    print(f"Moved {file_path.name} to other/")
                except Exception as e:
                    print(f"Error moving {file_path.name}: {e}")
    
    print("File organization completed.")

def main():
    import sys
    
    if len(sys.argv) > 1:
        source_directory = sys.argv[1]
        target_directory = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        source_directory = input("Enter source directory path: ").strip()
        target_input = input("Enter target directory (press Enter to use source): ").strip()
        target_directory = target_input if target_input else None
    
    organize_files(source_directory, target_directory)

if __name__ == "__main__":
    main()