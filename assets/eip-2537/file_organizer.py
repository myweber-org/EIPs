
import os
import shutil

def organize_files(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            _, extension = os.path.splitext(filename)
            extension = extension.lower()
            
            if extension:
                folder_name = extension[1:] + "_files"
            else:
                folder_name = "no_extension_files"
            
            folder_path = os.path.join(directory, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            destination_path = os.path.join(folder_path, filename)
            shutil.move(file_path, destination_path)
            print(f"Moved {filename} to {folder_name}/")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)