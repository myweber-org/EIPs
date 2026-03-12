
import os
import shutil

def organize_files(directory):
    """
    Organize files in the specified directory into subdirectories based on their file extensions.
    """
    # Define categories and their associated file extensions
    categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'],
        'Documents': ['.pdf', '.docx', '.txt', '.xlsx', '.pptx', '.md'],
        'Archives': ['.zip', '.tar', '.gz', '.rar', '.7z'],
        'Audio': ['.mp3', '.wav', '.aac', '.flac'],
        'Video': ['.mp4', '.avi', '.mov', '.mkv', '.flv'],
        'Code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.json']
    }

    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    # Get all files in the directory
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    except PermissionError:
        print(f"Error: Permission denied when accessing '{directory}'.")
        return

    # Create category folders and move files
    for file in files:
        file_path = os.path.join(directory, file)
        file_ext = os.path.splitext(file)[1].lower()

        moved = False
        for category, extensions in categories.items():
            if file_ext in extensions:
                category_folder = os.path.join(directory, category)
                os.makedirs(category_folder, exist_ok=True)
                destination = os.path.join(category_folder, file)

                # Handle duplicate file names
                counter = 1
                base, ext = os.path.splitext(file)
                while os.path.exists(destination):
                    new_name = f"{base}_{counter}{ext}"
                    destination = os.path.join(category_folder, new_name)
                    counter += 1

                try:
                    shutil.move(file_path, destination)
                    print(f"Moved: {file} -> {category}/{os.path.basename(destination)}")
                except Exception as e:
                    print(f"Failed to move {file}: {e}")
                moved = True
                break

        if not moved:
            # Move uncategorized files to an 'Other' folder
            other_folder = os.path.join(directory, 'Other')
            os.makedirs(other_folder, exist_ok=True)
            destination = os.path.join(other_folder, file)

            counter = 1
            base, ext = os.path.splitext(file)
            while os.path.exists(destination):
                new_name = f"{base}_{counter}{ext}"
                destination = os.path.join(other_folder, new_name)
                counter += 1

            try:
                shutil.move(file_path, destination)
                print(f"Moved: {file} -> Other/{os.path.basename(destination)}")
            except Exception as e:
                print(f"Failed to move {file}: {e}")

    print("File organization complete.")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to organize: ").strip()
    organize_files(target_directory)