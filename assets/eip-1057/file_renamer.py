
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
import os
import sys

def rename_files_with_sequential_numbers(directory_path, prefix="file", start_number=1):
    """
    Rename all files in the specified directory with sequential numbering.
    
    Args:
        directory_path (str): Path to the directory containing files to rename.
        prefix (str): Prefix to use for renamed files. Default is "file".
        start_number (int): Starting number for sequential naming. Default is 1.
    
    Returns:
        dict: Dictionary mapping old filenames to new filenames.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return {}
    
    try:
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        files.sort()
        
        renamed_files = {}
        counter = start_number
        
        for filename in files:
            file_extension = os.path.splitext(filename)[1]
            new_filename = f"{prefix}_{counter:03d}{file_extension}"
            old_path = os.path.join(directory_path, filename)
            new_path = os.path.join(directory_path, new_filename)
            
            os.rename(old_path, new_path)
            renamed_files[filename] = new_filename
            counter += 1
        
        print(f"Successfully renamed {len(renamed_files)} files.")
        for old_name, new_name in renamed_files.items():
            print(f"  {old_name} -> {new_name}")
        
        return renamed_files
        
    except Exception as e:
        print(f"Error during renaming: {e}")
        return {}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <directory_path> [prefix] [start_number]")
        sys.exit(1)
    
    dir_path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "file"
    start_num = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    rename_files_with_sequential_numbers(dir_path, prefix, start_num)