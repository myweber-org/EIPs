
import os
import sys

def rename_files(directory, prefix="file"):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files.sort()
        
        for index, filename in enumerate(files, start=1):
            extension = os.path.splitext(filename)[1]
            new_name = f"{prefix}_{index:03d}{extension}"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
            
        print(f"Successfully renamed {len(files)} files.")
        
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
    except PermissionError:
        print(f"Error: Permission denied for directory '{directory}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory> [prefix]")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    name_prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    
    rename_files(target_dir, name_prefix)