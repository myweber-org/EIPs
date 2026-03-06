
import sys
import os

def remove_duplicate_lines(input_file, output_file=None):
    """
    Remove duplicate lines from a file while preserving order.
    
    Args:
        input_file (str): Path to the input file.
        output_file (str, optional): Path to the output file. 
                                     If None, overwrites input file.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
    
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        seen = set()
        unique_lines = []
        
        for line in lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        
        if output_file is None:
            output_file = input_file
        
        with open(output_file, 'w') as f:
            f.writelines(unique_lines)
        
        removed_count = len(lines) - len(unique_lines)
        print(f"Removed {removed_count} duplicate lines.")
        print(f"Original: {len(lines)} lines, New: {len(unique_lines)} lines.")
        
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_cleaner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicate_lines(input_file, output_file)