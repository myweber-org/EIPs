
import os
import shutil
import tempfile
from pathlib import Path

def clean_temporary_files(directory_path, extensions=('.tmp', '.temp', '.log', '.cache')):
    """
    Remove temporary files with specified extensions from a directory.
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"Provided path is not a directory: {directory_path}")

    cleaned_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(extensions):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    cleaned_files.append(file_path)
                    print(f"Removed: {file_path}")
                except OSError as e:
                    print(f"Error removing {file_path}: {e}")

    return cleaned_files

def create_test_environment():
    """
    Create a test directory with temporary files for demonstration.
    """
    test_dir = tempfile.mkdtemp(prefix="clean_test_")
    print(f"Created test directory: {test_dir}")

    test_files = [
        "document.tmp",
        "image.cache",
        "backup.log",
        "data.temp",
        "normal.txt",
        "important.doc"
    ]

    for fname in test_files:
        file_path = Path(test_dir) / fname
        file_path.touch()
        print(f"Created: {file_path}")

    return test_dir

if __name__ == "__main__":
    try:
        test_dir = create_test_environment()
        print("\n--- Cleaning temporary files ---")
        removed = clean_temporary_files(test_dir)
        print(f"\nTotal files removed: {len(removed)}")

        print("\n--- Remaining files ---")
        for item in Path(test_dir).iterdir():
            print(f"  {item.name}")

        shutil.rmtree(test_dir)
        print(f"\nCleaned up test directory: {test_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")