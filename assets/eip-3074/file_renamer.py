
import os
import sys
import argparse

def rename_files(directory, prefix, start_number=1, extension=None):
    """
    Rename files in a directory with sequential numbering.
    
    Args:
        directory (str): Path to the directory containing files
        prefix (str): Prefix for renamed files
        start_number (int): Starting number for sequential naming
        extension (str): Filter files by extension (optional)
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return False
    
    try:
        files = os.listdir(directory)
        
        if extension:
            files = [f for f in files if f.lower().endswith(f'.{extension.lower()}')]
        
        files.sort()
        
        for index, filename in enumerate(files, start=start_number):
            old_path = os.path.join(directory, filename)
            
            if extension:
                new_filename = f"{prefix}_{index:03d}.{extension}"
            else:
                file_ext = os.path.splitext(filename)[1]
                new_filename = f"{prefix}_{index:03d}{file_ext}"
            
            new_path = os.path.join(directory, new_filename)
            
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
            except OSError as e:
                print(f"Error renaming {filename}: {e}")
        
        print(f"Successfully renamed {len(files)} files.")
        return True
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Rename files with sequential numbering.')
    parser.add_argument('directory', help='Directory containing files to rename')
    parser.add_argument('prefix', help='Prefix for renamed files')
    parser.add_argument('--start', type=int, default=1, help='Starting number (default: 1)')
    parser.add_argument('--ext', help='File extension filter (e.g., jpg, png)')
    
    args = parser.parse_args()
    
    success = rename_files(args.directory, args.prefix, args.start, args.ext)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()