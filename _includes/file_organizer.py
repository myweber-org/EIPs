
import os
import shutil
from pathlib import Path

def organize_files_by_extension(directory_path):
    """
    Organize files in the specified directory by moving them into subfolders
    based on their file extensions.
    """
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    path = Path(directory_path)
    
    # Define categories and their associated file extensions
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Audio': ['.mp3', '.wav', '.flac', '.aac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv'],
        'Archives': ['.zip', '.tar', '.gz', '.rar'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp'],
        'Executables': ['.exe', '.msi', '.sh', '.bat']
    }
    
    # Create category folders if they don't exist
    for category in categories:
        category_path = path / category
        category_path.mkdir(exist_ok=True)
    
    # Track moved files
    moved_files = []
    skipped_files = []
    
    # Iterate through files in the directory
    for item in path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            moved = False
            
            # Find the appropriate category for the file
            for category, extensions in categories.items():
                if file_extension in extensions:
                    destination = path / category / item.name
                    
                    # Handle duplicate filenames
                    if destination.exists():
                        base_name = item.stem
                        counter = 1
                        while destination.exists():
                            new_name = f"{base_name}_{counter}{item.suffix}"
                            destination = path / category / new_name
                            counter += 1
                    
                    try:
                        shutil.move(str(item), str(destination))
                        moved_files.append((item.name, category))
                        moved = True
                        break
                    except Exception as e:
                        print(f"Error moving file {item.name}: {e}")
                        skipped_files.append(item.name)
            
            # If file doesn't match any category, move to 'Other'
            if not moved:
                other_folder = path / 'Other'
                other_folder.mkdir(exist_ok=True)
                destination = other_folder / item.name
                
                # Handle duplicate filenames in Other folder
                if destination.exists():
                    base_name = item.stem
                    counter = 1
                    while destination.exists():
                        new_name = f"{base_name}_{counter}{item.suffix}"
                        destination = other_folder / new_name
                        counter += 1
                
                try:
                    shutil.move(str(item), str(destination))
                    moved_files.append((item.name, 'Other'))
                except Exception as e:
                    print(f"Error moving file {item.name}: {e}")
                    skipped_files.append(item.name)
    
    # Print summary
    print(f"\nOrganization complete for: {directory_path}")
    print(f"Total files moved: {len(moved_files)}")
    
    if moved_files:
        print("\nFiles moved:")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    
    if skipped_files:
        print(f"\nFiles skipped (could not move): {len(skipped_files)}")
        for filename in skipped_files:
            print(f"  {filename}")

if __name__ == "__main__":
    # Example usage: organize files in the current directory
    current_directory = os.getcwd()
    organize_files_by_extension(current_directory)