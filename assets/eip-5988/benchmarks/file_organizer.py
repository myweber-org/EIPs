
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the given directory by moving them into subfolders
    based on their file extensions.
    """
    # Define file type categories and their associated extensions
    file_categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Audio': ['.mp3', '.wav', '.aac', '.flac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.rar', '.tar', '.gz'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp']
    }
    
    # Ensure the directory exists
    target_dir = Path(directory)
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Error: Directory '{directory}' does not exist or is not a valid directory.")
        return
    
    # Create category folders if they don't exist
    for category in file_categories.keys():
        category_path = target_dir / category
        category_path.mkdir(exist_ok=True)
    
    # Track moved files and errors
    moved_files = []
    error_files = []
    
    # Iterate over files in the directory
    for item in target_dir.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            moved = False
            
            # Find the appropriate category for the file
            for category, extensions in file_categories.items():
                if file_extension in extensions:
                    destination = target_dir / category / item.name
                    try:
                        # Check if file already exists in destination
                        if destination.exists():
                            base_name = item.stem
                            counter = 1
                            while destination.exists():
                                new_name = f"{base_name}_{counter}{item.suffix}"
                                destination = target_dir / category / new_name
                                counter += 1
                        
                        shutil.move(str(item), str(destination))
                        moved_files.append((item.name, category))
                        moved = True
                        break
                    except Exception as e:
                        error_files.append((item.name, str(e)))
                        moved = True
                        break
            
            # If file doesn't match any category, move to 'Other'
            if not moved:
                other_dir = target_dir / 'Other'
                other_dir.mkdir(exist_ok=True)
                destination = other_dir / item.name
                try:
                    shutil.move(str(item), str(destination))
                    moved_files.append((item.name, 'Other'))
                except Exception as e:
                    error_files.append((item.name, str(e)))
    
    # Print summary
    print(f"Organization complete for directory: {directory}")
    print(f"Total files processed: {len(moved_files) + len(error_files)}")
    print(f"Successfully moved: {len(moved_files)}")
    print(f"Errors encountered: {len(error_files)}")
    
    if moved_files:
        print("\nMoved files:")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    
    if error_files:
        print("\nFiles with errors:")
        for filename, error in error_files:
            print(f"  {filename}: {error}")

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    current_directory = os.getcwd()
    organize_files(current_directory)