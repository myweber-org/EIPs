
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