
import os
import datetime

def rename_files_in_directory(directory_path, prefix="file_"):
    try:
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        
        for filename in files:
            file_path = os.path.join(directory_path, filename)
            creation_time = os.path.getctime(file_path)
            date_str = datetime.datetime.fromtimestamp(creation_time).strftime("%Y%m%d_%H%M%S")
            
            file_extension = os.path.splitext(filename)[1]
            new_filename = f"{prefix}{date_str}{file_extension}"
            new_file_path = os.path.join(directory_path, new_filename)
            
            counter = 1
            while os.path.exists(new_file_path):
                new_filename = f"{prefix}{date_str}_{counter}{file_extension}"
                new_file_path = os.path.join(directory_path, new_filename)
                counter += 1
            
            os.rename(file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")
            
        print("File renaming completed successfully.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path: ").strip()
    custom_prefix = input("Enter file prefix (press Enter for default): ").strip()
    
    if not custom_prefix:
        custom_prefix = "file_"
    
    if os.path.isdir(target_directory):
        rename_files_in_directory(target_directory, custom_prefix)
    else:
        print("Invalid directory path.")