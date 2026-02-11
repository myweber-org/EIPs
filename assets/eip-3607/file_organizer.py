
import os
import shutil

def organize_files(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_extension = filename.split('.')[-1] if '.' in filename else 'no_extension'
            target_folder = os.path.join(directory, file_extension.upper())
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            shutil.move(file_path, os.path.join(target_folder, filename))
            print(f"Moved {filename} to {file_extension.upper()}/")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ")
    if os.path.isdir(target_directory):
        organize_files(target_directory)
        print("File organization completed.")
    else:
        print("Invalid directory path.")