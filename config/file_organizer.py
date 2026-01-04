
import os
import shutil

def organize_files(directory):
    """
    Organize files in the specified directory by their extensions.
    Creates subdirectories for each file type and moves files accordingly.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            file_extension = filename.split('.')[-1] if '.' in filename else 'no_extension'
            target_dir = os.path.join(directory, file_extension)
            
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            try:
                shutil.move(file_path, os.path.join(target_dir, filename))
                print(f"Moved: {filename} -> {file_extension}/")
            except Exception as e:
                print(f"Failed to move {filename}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter directory path to organize: ").strip()
    organize_files(target_directory)