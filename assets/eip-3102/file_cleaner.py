
import os
import glob
import sys

def clean_temp_files(directory, extensions=('.tmp', '.temp', '.log')):
    """
    Remove temporary files with specified extensions from a directory.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return False
    
    removed_files = []
    for ext in extensions:
        pattern = os.path.join(directory, f'*{ext}')
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                removed_files.append(file_path)
                print(f"Removed: {file_path}")
            except OSError as e:
                print(f"Error removing {file_path}: {e}")
    
    if removed_files:
        print(f"Cleaned up {len(removed_files)} temporary files.")
    else:
        print("No temporary files found to clean.")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = os.getcwd()
    
    clean_temp_files(target_dir)