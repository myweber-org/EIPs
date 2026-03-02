
import os
import shutil
from pathlib import Path

def organize_files(directory):
    """
    Organizes files in the specified directory by moving them into
    subfolders named after their file extensions.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        if os.path.isfile(item_path):
            file_extension = Path(item).suffix.lower()

            if not file_extension:
                folder_name = "no_extension"
            else:
                folder_name = file_extension[1:]

            target_folder = os.path.join(directory, folder_name)
            os.makedirs(target_folder, exist_ok=True)

            try:
                shutil.move(item_path, os.path.join(target_folder, item))
                print(f"Moved: {item} -> {folder_name}/")
            except Exception as e:
                print(f"Failed to move {item}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)
import os
import shutil
from datetime import datetime
from pathlib import Path
import hashlib

def calculate_file_hash(filepath):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def organize_files_by_date(source_dir, target_dir):
    """
    Organize files from source directory to target directory 
    based on creation/modification date.
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"Source directory {source_dir} does not exist.")
        return
    
    target_path.mkdir(parents=True, exist_ok=True)
    
    file_hashes = set()
    organized_count = 0
    duplicate_count = 0
    
    for file_path in source_path.rglob("*"):
        if file_path.is_file():
            try:
                # Get file modification time
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                year_month = mod_time.strftime("%Y-%m")
                
                # Create year-month directory in target
                month_dir = target_path / year_month
                month_dir.mkdir(exist_ok=True)
                
                # Calculate file hash for duplicate detection
                file_hash = calculate_file_hash(file_path)
                
                if file_hash in file_hashes:
                    print(f"Duplicate found: {file_path.name}")
                    duplicate_count += 1
                    continue
                
                file_hashes.add(file_hash)
                
                # Determine new filename (preserve extension)
                new_filename = file_path.name
                new_filepath = month_dir / new_filename
                
                # Handle filename conflicts
                counter = 1
                while new_filepath.exists():
                    name_parts = file_path.stem.split('_')
                    if name_parts[-1].isdigit():
                        base_name = '_'.join(name_parts[:-1])
                    else:
                        base_name = file_path.stem
                    
                    new_filename = f"{base_name}_{counter}{file_path.suffix}"
                    new_filepath = month_dir / new_filename
                    counter += 1
                
                # Move file to organized location
                shutil.move(str(file_path), str(new_filepath))
                organized_count += 1
                
                print(f"Moved: {file_path.name} -> {year_month}/{new_filename}")
                
            except PermissionError:
                print(f"Permission denied: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    print(f"\nOrganization complete!")
    print(f"Files organized: {organized_count}")
    print(f"Duplicates skipped: {duplicate_count}")

def create_sample_files(test_dir, num_files=10):
    """Create sample files for testing."""
    test_path = Path(test_dir)
    test_path.mkdir(exist_ok=True)
    
    for i in range(num_files):
        filename = test_path / f"document_{i}.txt"
        with open(filename, 'w') as f:
            f.write(f"This is sample file {i}\nCreated for testing purposes.")
        
        # Modify creation time to simulate different dates
        random_days = i * 7
        random_time = datetime.now().timestamp() - (random_days * 86400)
        os.utime(filename, (random_time, random_time))

if __name__ == "__main__":
    # Example usage
    source_directory = "./test_source"
    target_directory = "./organized_files"
    
    # Create test files if source doesn't exist
    if not Path(source_directory).exists():
        print("Creating sample files for demonstration...")
        create_sample_files(source_directory, 15)
    
    # Organize files
    organize_files_by_date(source_directory, target_directory)