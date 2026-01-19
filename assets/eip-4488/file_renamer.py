import os
import sys

def rename_files_sequentially(directory, prefix="file", extension=".txt"):
    """
    Rename all files in the given directory with sequential numbering.
    
    Args:
        directory (str): Path to the directory containing files to rename
        prefix (str): Prefix for renamed files
        extension (str): File extension to filter and apply
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    files = [f for f in os.listdir(directory) 
             if os.path.isfile(os.path.join(directory, f)) and f.endswith(extension)]
    files.sort()
    
    if not files:
        print(f"No files with extension '{extension}' found in '{directory}'.")
        return
    
    renamed_count = 0
    for index, filename in enumerate(files, start=1):
        old_path = os.path.join(directory, filename)
        new_name = f"{prefix}_{index:03d}{extension}"
        new_path = os.path.join(directory, new_name)
        
        if old_path != new_path:
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_name}")
                renamed_count += 1
            except OSError as e:
                print(f"Error renaming {filename}: {e}")
    
    print(f"\nRenamed {renamed_count} file(s) successfully.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory> [prefix] [extension]")
        print("Example: python file_renamer.py ./documents document .pdf")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    file_prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    file_extension = sys.argv[3] if len(sys.argv) > 3 else ".txt"
    
    rename_files_sequentially(dir_path, file_prefix, file_extension)