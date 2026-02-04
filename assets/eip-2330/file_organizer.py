
import os
import shutil

def organize_files(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            file_extension = filename.split('.')[-1] if '.' in filename else 'no_extension'
            target_dir = os.path.join(directory, file_extension)
            
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            shutil.move(file_path, os.path.join(target_dir, filename))
            print(f"Moved {filename} to {file_extension}/")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ")
    organize_files(target_directory)