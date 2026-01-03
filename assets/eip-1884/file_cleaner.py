
import sys
import os

def remove_duplicates(input_file, output_file=None):
    """
    Remove duplicate lines from a text file while preserving order.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
    
    if output_file is None:
        output_file = input_file + ".deduped"
    
    seen_lines = set()
    unique_lines = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.rstrip('\n')
                if stripped_line not in seen_lines:
                    seen_lines.add(stripped_line)
                    unique_lines.append(line)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(unique_lines)
        
        print(f"Successfully removed duplicates. Output saved to: {output_file}")
        print(f"Original lines: {len(seen_lines) + (len(unique_lines) - len(seen_lines))}")
        print(f"Unique lines: {len(unique_lines)}")
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_cleaner.py <input_file> [output_file]")
        print("Example: python file_cleaner.py data.txt cleaned_data.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)