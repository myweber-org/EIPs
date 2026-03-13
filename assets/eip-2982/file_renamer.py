import os
import sys

def rename_files_with_sequence(directory, prefix="file", extension=".txt"):
    """
    Rename all files in the specified directory with sequential numbering.
    
    Args:
        directory (str): Path to the directory containing files to rename.
        prefix (str): Prefix for renamed files.
        extension (str): File extension to filter and apply.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    files = [f for f in os.listdir(directory) if f.endswith(extension)]
    files.sort()
    
    if not files:
        print(f"No files with extension '{extension}' found in '{directory}'.")
        return
    
    for index, filename in enumerate(files, start=1):
        old_path = os.path.join(directory, filename)
        new_filename = f"{prefix}_{index:03d}{extension}"
        new_path = os.path.join(directory, new_filename)
        
        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
        except OSError as e:
            print(f"Failed to rename {filename}: {e}")
    
    print(f"Renaming complete. {len(files)} files processed.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory> [prefix] [extension]")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    file_prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    file_extension = sys.argv[3] if len(sys.argv) > 3 else ".txt"
    
    rename_files_with_sequence(dir_path, file_prefix, file_extension)
import os
import sys

def batch_rename_files(directory, prefix, extension):
    """
    Rename all files in a directory with sequential numbering.
    
    Args:
        directory (str): Path to the directory containing files
        prefix (str): Prefix for the new filenames
        extension (str): File extension (without dot)
    
    Returns:
        int: Number of files renamed
    """
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return 0
    
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a directory.")
        return 0
    
    files = [f for f in os.listdir(directory) 
             if os.path.isfile(os.path.join(directory, f))]
    
    if not files:
        print("No files found in the directory.")
        return 0
    
    renamed_count = 0
    
    for index, filename in enumerate(files, start=1):
        old_path = os.path.join(directory, filename)
        
        file_ext = os.path.splitext(filename)[1]
        if not file_ext:
            file_ext = f".{extension}"
        
        new_filename = f"{prefix}_{index:03d}{file_ext}"
        new_path = os.path.join(directory, new_filename)
        
        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
            renamed_count += 1
        except OSError as e:
            print(f"Failed to rename {filename}: {e}")
    
    print(f"\nTotal files renamed: {renamed_count}")
    return renamed_count

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python file_renamer.py <directory> <prefix> <extension>")
        print("Example: python file_renamer.py ./photos vacation jpg")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    name_prefix = sys.argv[2]
    file_extension = sys.argv[3]
    
    batch_rename_files(target_dir, name_prefix, file_extension)