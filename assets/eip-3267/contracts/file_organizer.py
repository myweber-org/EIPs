
import os
import shutil
from pathlib import Path

def organize_files(directory="."):
    """
    Organizes files in the specified directory by moving them into
    subfolders based on their file extensions.
    """
    base_path = Path(directory).resolve()

    # Define categories and their associated file extensions
    categories = {
        "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg"],
        "Documents": [".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md"],
        "Audio": [".mp3", ".wav", ".flac", ".aac"],
        "Video": [".mp4", ".avi", ".mov", ".mkv"],
        "Archives": [".zip", ".tar", ".gz", ".rar", ".7z"],
        "Code": [".py", ".js", ".html", ".css", ".java", ".cpp", ".c"],
    }

    # Create category folders if they don't exist
    for category in categories:
        (base_path / category).mkdir(exist_ok=True)

    # Track moved files
    moved_files = []

    for item in base_path.iterdir():
        if item.is_file():
            file_extension = item.suffix.lower()
            moved = False

            # Find the appropriate category for the file
            for category, extensions in categories.items():
                if file_extension in extensions:
                    target_folder = base_path / category
                    try:
                        shutil.move(str(item), str(target_folder / item.name))
                        moved_files.append((item.name, category))
                        moved = True
                        break
                    except Exception as e:
                        print(f"Error moving {item.name}: {e}")

            # If no category matched, move to 'Other'
            if not moved:
                other_folder = base_path / "Other"
                other_folder.mkdir(exist_ok=True)
                try:
                    shutil.move(str(item), str(other_folder / item.name))
                    moved_files.append((item.name, "Other"))
                except Exception as e:
                    print(f"Error moving {item.name}: {e}")

    # Print summary
    if moved_files:
        print(f"Organized {len(moved_files)} file(s) in '{base_path}':")
        for filename, category in moved_files:
            print(f"  {filename} -> {category}/")
    else:
        print("No files were moved.")

if __name__ == "__main__":
    organize_files()