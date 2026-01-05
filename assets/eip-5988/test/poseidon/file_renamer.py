
import os
import sys

def rename_files_in_directory(directory_path, prefix):
    """
    Rename all files in the specified directory by adding a prefix.
    """
    try:
        if not os.path.isdir(directory_path):
            print(f"Error: {directory_path} is not a valid directory.")
            return False

        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        
        if not files:
            print("No files found in the directory.")
            return True

        for filename in files:
            new_name = f"{prefix}_{filename}"
            old_path = os.path.join(directory_path, filename)
            new_path = os.path.join(directory_path, new_name)
            
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_name}")
            except OSError as e:
                print(f"Failed to rename {filename}: {e}")
        
        print("Renaming completed.")
        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python file_renamer.py <directory_path> <prefix>")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    file_prefix = sys.argv[2]
    
    success = rename_files_in_directory(dir_path, file_prefix)
    sys.exit(0 if success else 1)