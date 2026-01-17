
import sys

def remove_duplicate_lines(input_file, output_file=None):
    """
    Remove duplicate lines from a file while preserving the order of first occurrence.
    
    Args:
        input_file (str): Path to the input file.
        output_file (str, optional): Path to the output file. 
                                     If None, overwrites the input file.
    """
    if output_file is None:
        output_file = input_file
    
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        seen = set()
        unique_lines = []
        
        for line in lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        
        with open(output_file, 'w') as f:
            f.writelines(unique_lines)
            
        print(f"Removed {len(lines) - len(unique_lines)} duplicate lines.")
        print(f"Original: {len(lines)} lines, After: {len(unique_lines)} lines.")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except IOError as e:
        print(f"Error: Unable to process file - {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_cleaner.py <input_file> [output_file]")
        print("If output_file is not provided, input_file will be overwritten.")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicate_lines(input_file, output_file)