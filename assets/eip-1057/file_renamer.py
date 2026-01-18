
import os
import sys
import argparse

def rename_files(directory, prefix='file', start_number=1, extension='.txt'):
    """
    Rename all files in a directory with sequential numbering.
    
    Args:
        directory (str): Path to the directory containing files to rename
        prefix (str): Prefix for renamed files
        start_number (int): Starting number for sequential naming
        extension (str): File extension to filter and apply
    """
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a directory.")
        return
    
    files = [f for f in os.listdir(directory) 
             if os.path.isfile(os.path.join(directory, f)) and f.endswith(extension)]
    
    if not files:
        print(f"No files with extension '{extension}' found in '{directory}'.")
        return
    
    files.sort()
    
    renamed_count = 0
    current_number = start_number
    
    for filename in files:
        old_path = os.path.join(directory, filename)
        new_filename = f"{prefix}_{current_number:03d}{extension}"
        new_path = os.path.join(directory, new_filename)
        
        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
            renamed_count += 1
            current_number += 1
        except OSError as e:
            print(f"Error renaming {filename}: {e}")
    
    print(f"\nRenamed {renamed_count} file(s) successfully.")

def main():
    parser = argparse.ArgumentParser(description='Rename files with sequential numbering.')
    parser.add_argument('directory', help='Directory containing files to rename')
    parser.add_argument('--prefix', default='file', help='Prefix for renamed files (default: file)')
    parser.add_argument('--start', type=int, default=1, help='Starting number (default: 1)')
    parser.add_argument('--ext', default='.txt', help='File extension to process (default: .txt)')
    
    args = parser.parse_args()
    
    rename_files(args.directory, args.prefix, args.start, args.ext)

if __name__ == '__main__':
    main()