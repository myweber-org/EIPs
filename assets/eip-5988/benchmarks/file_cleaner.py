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

    removed_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(extensions):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    removed_files.append(file_path)
                    print(f"Removed: {file_path}")
                except OSError as e:
                    print(f"Error removing {file_path}: {e}")

    return removed_files

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
        "important.txt",
        "data.temp",
        "config.cache"
    ]

    for file_name in test_files:
        file_path = Path(test_dir) / file_name
        file_path.touch()
        print(f"Created: {file_path}")

    return test_dir

if __name__ == "__main__":
    # Create a test environment and clean it
    test_directory = create_test_environment()
    print("\n--- Cleaning temporary files ---")
    cleaned = clean_temporary_files(test_directory)

    print(f"\nTotal files removed: {len(cleaned)}")

    # Clean up the test directory itself
    shutil.rmtree(test_directory)
    print(f"Removed test directory: {test_directory}")