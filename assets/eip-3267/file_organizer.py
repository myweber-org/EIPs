
import os
import shutil
from pathlib import Path

def organize_files(source_dir, organize_by='extension'):
    """
    Organize files in a directory by moving them into subfolders based on their extension.
    
    Args:
        source_dir (str): Path to the directory containing files to organize
        organize_by (str): Organization method - 'extension' or 'type'
    """
    if not os.path.exists(source_dir):
        print(f"Source directory does not exist: {source_dir}")
        return
    
    source_path = Path(source_dir)
    
    # Define category mappings for file types
    type_categories = {
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp'],
        'documents': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', '.xls', '.xlsx'],
        'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg'],
        'video': ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv'],
        'archives': ['.zip', '.rar', '.7z', '.tar', '.gz'],
        'code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.php'],
        'data': ['.csv', '.json', '.xml', '.sql', '.db']
    }
    
    # Process each file in the source directory
    for item in source_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            
            if organize_by == 'extension':
                # Create folder named after the extension (without dot)
                folder_name = file_extension[1:] if file_extension else 'no_extension'
                target_folder = source_path / folder_name
            else:
                # Find the category for this file type
                folder_name = 'other'
                for category, extensions in type_categories.items():
                    if file_extension in extensions:
                        folder_name = category
                        break
                target_folder = source_path / folder_name
            
            # Create target folder if it doesn't exist
            target_folder.mkdir(exist_ok=True)
            
            # Move the file
            try:
                shutil.move(str(item), str(target_folder / item.name))
                print(f"Moved: {item.name} -> {folder_name}/")
            except Exception as e:
                print(f"Error moving {item.name}: {e}")

def main():
    # Example usage
    current_directory = os.getcwd()
    print(f"Organizing files in: {current_directory}")
    
    # Organize by file type
    organize_files(current_directory, organize_by='type')
    
    print("File organization completed.")

if __name__ == "__main__":
    main()