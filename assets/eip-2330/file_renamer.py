
import os
import sys
import argparse

def rename_files(directory, prefix, start_number=1, extension=None):
    """
    Rename files in a directory with sequential numbering.
    
    Args:
        directory (str): Path to the directory containing files
        prefix (str): Prefix for renamed files
        start_number (int): Starting number for sequencing
        extension (str): Filter files by extension (optional)
    """
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a directory.")
        return
    
    files = os.listdir(directory)
    
    if extension:
        files = [f for f in files if f.lower().endswith(f".{extension.lower()}")]
    
    if not files:
        print(f"No files found in '{directory}'" + 
              (f" with extension '.{extension}'" if extension else ""))
        return
    
    files.sort()
    
    counter = start_number
    
    for filename in files:
        file_extension = os.path.splitext(filename)[1]
        new_filename = f"{prefix}_{counter:03d}{file_extension}"
        
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        
        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
            counter += 1
        except Exception as e:
            print(f"Failed to rename {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Rename files in a directory with sequential numbering."
    )
    
    parser.add_argument(
        "directory",
        help="Directory containing files to rename"
    )
    
    parser.add_argument(
        "prefix",
        help="Prefix for renamed files"
    )
    
    parser.add_argument(
        "-s", "--start",
        type=int,
        default=1,
        help="Starting number (default: 1)"
    )
    
    parser.add_argument(
        "-e", "--extension",
        help="Filter files by extension (e.g., 'jpg', 'png')"
    )
    
    args = parser.parse_args()
    
    rename_files(
        directory=args.directory,
        prefix=args.prefix,
        start_number=args.start,
        extension=args.extension
    )

if __name__ == "__main__":
    main()