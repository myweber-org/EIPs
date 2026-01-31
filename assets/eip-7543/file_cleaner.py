
import os
import shutil
import tempfile
from pathlib import Path

def clean_temporary_files(directory_path, extensions=None, days_old=7):
    """
    Remove temporary files from a specified directory.
    
    Args:
        directory_path: Path to the directory to clean.
        extensions: List of file extensions to consider as temporary.
                   If None, uses common temporary extensions.
        days_old: Remove files older than this many days.
    """
    if extensions is None:
        extensions = ['.tmp', '.temp', '.log', '.cache', '.bak']
    
    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"Directory {directory_path} does not exist")
    
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory_path} is not a directory")
    
    current_time = time.time()
    cutoff_time = current_time - (days_old * 24 * 60 * 60)
    
    removed_files = []
    removed_size = 0
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            file_age = file_path.stat().st_mtime
            
            if file_age < cutoff_time or file_path.suffix.lower() in extensions:
                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    removed_files.append(str(file_path))
                    removed_size += file_size
                except (PermissionError, OSError) as e:
                    print(f"Could not remove {file_path}: {e}")
    
    return {
        'removed_files': removed_files,
        'removed_count': len(removed_files),
        'removed_size_bytes': removed_size
    }

def create_sample_temp_files(test_dir):
    """Create sample temporary files for testing."""
    test_dir = Path(test_dir)
    test_dir.mkdir(exist_ok=True)
    
    sample_files = [
        test_dir / 'document.tmp',
        test_dir / 'backup.bak',
        test_dir / 'application.log',
        test_dir / 'data.cache',
        test_dir / 'normal.txt'
    ]
    
    for file_path in sample_files:
        file_path.write_text('Sample content for testing')
    
    return sample_files

if __name__ == "__main__":
    import time
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Creating test files in: {temp_dir}")
        sample_files = create_sample_temp_files(temp_dir)
        
        time.sleep(1)
        
        result = clean_temporary_files(temp_dir, days_old=0)
        
        print(f"Removed {result['removed_count']} files")
        print(f"Freed {result['removed_size_bytes']} bytes")
        
        remaining_files = list(Path(temp_dir).glob('*'))
        print(f"Remaining files: {[f.name for f in remaining_files]}")