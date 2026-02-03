
import os
import datetime
import sys

def add_timestamp_prefix(filepath):
    if not os.path.exists(filepath):
        return f"Error: File '{filepath}' does not exist"
    
    directory, filename = os.path.split(filepath)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{timestamp}_{filename}"
    new_filepath = os.path.join(directory, new_filename)
    
    try:
        os.rename(filepath, new_filepath)
        return f"Renamed '{filename}' to '{new_filename}'"
    except Exception as e:
        return f"Error renaming file: {str(e)}"

def process_files(file_list):
    results = []
    for filepath in file_list:
        result = add_timestamp_prefix(filepath)
        results.append(result)
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_renamer.py <file1> [file2 ...]")
        sys.exit(1)
    
    files = sys.argv[1:]
    output = process_files(files)
    
    for line in output:
        print(line)
import os
import sys
import argparse

def rename_files(directory, prefix, start_number=1, extension=None):
    """
    Rename files in a directory with sequential numbering.
    
    Args:
        directory (str): Path to directory containing files
        prefix (str): Prefix for renamed files
        start_number (int): Starting number for sequence
        extension (str): Filter files by extension (optional)
    """
    try:
        if not os.path.isdir(directory):
            print(f"Error: Directory '{directory}' does not exist.")
            return False
        
        files = os.listdir(directory)
        
        if extension:
            files = [f for f in files if f.lower().endswith(f'.{extension.lower()}')]
        
        files.sort()
        
        counter = start_number
        
        for filename in files:
            old_path = os.path.join(directory, filename)
            
            if os.path.isfile(old_path):
                file_ext = os.path.splitext(filename)[1]
                new_filename = f"{prefix}_{counter:03d}{file_ext}"
                new_path = os.path.join(directory, new_filename)
                
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} -> {new_filename}")
                    counter += 1
                except Exception as e:
                    print(f"Failed to rename {filename}: {e}")
        
        print(f"Renaming complete. {counter - start_number} files renamed.")
        return True
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Rename files with sequential numbering.')
    parser.add_argument('directory', help='Directory containing files to rename')
    parser.add_argument('prefix', help='Prefix for renamed files')
    parser.add_argument('--start', type=int, default=1, help='Starting number (default: 1)')
    parser.add_argument('--ext', help='Filter by file extension (e.g., jpg, png)')
    
    args = parser.parse_args()
    
    success = rename_files(args.directory, args.prefix, args.start, args.ext)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()