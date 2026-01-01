
import os
import sys
from datetime import datetime

def rename_files_by_date(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return False
    
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    for filename in files:
        filepath = os.path.join(directory, filename)
        stat = os.stat(filepath)
        creation_time = datetime.fromtimestamp(stat.st_ctime)
        new_name = creation_time.strftime("%Y%m%d_%H%M%S") + os.path.splitext(filename)[1]
        new_path = os.path.join(directory, new_name)
        
        counter = 1
        while os.path.exists(new_path):
            name_part, ext_part = os.path.splitext(new_name)
            new_name = f"{name_part}_{counter}{ext_part}"
            new_path = os.path.join(directory, new_name)
            counter += 1
        
        try:
            os.rename(filepath, new_path)
            print(f"Renamed: {filename} -> {new_name}")
        except Exception as e:
            print(f"Failed to rename {filename}: {e}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python file_renamer.py <directory_path>")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    rename_files_by_date(target_dir)
import os
import datetime

def add_timestamp_to_filename(filepath):
    """
    Rename a file by adding a timestamp prefix to its filename.
    """
    if not os.path.exists(filepath):
        return f"Error: File '{filepath}' does not exist."

    directory, filename = os.path.split(filepath)
    name, extension = os.path.splitext(filename)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{timestamp}_{name}{extension}"
    new_filepath = os.path.join(directory, new_filename)

    try:
        os.rename(filepath, new_filepath)
        return f"Renamed '{filename}' to '{new_filename}'"
    except Exception as e:
        return f"Error renaming file: {e}"

def process_directory(directory_path):
    """
    Add timestamp prefix to all files in the specified directory.
    """
    if not os.path.isdir(directory_path):
        return f"Error: '{directory_path}' is not a valid directory."

    results = []
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isfile(item_path):
            result = add_timestamp_to_filename(item_path)
            results.append(result)

    return results

if __name__ == "__main__":
    sample_file = "example_document.txt"
    with open(sample_file, 'w') as f:
        f.write("Sample content for testing.")

    print(add_timestamp_to_filename(sample_file))

    test_dir = "test_directory"
    os.makedirs(test_dir, exist_ok=True)
    for i in range(3):
        with open(os.path.join(test_dir, f"file_{i}.txt"), 'w') as f:
            f.write(f"Content of file {i}")

    dir_results = process_directory(test_dir)
    for res in dir_results:
        print(res)