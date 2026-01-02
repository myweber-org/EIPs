import sys

def remove_duplicates(input_file, output_file):
    seen = set()
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    
    with open(output_file, 'w') as outfile:
        for line in lines:
            if line not in seen:
                outfile.write(line)
                seen.add(line)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python file_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        remove_duplicates(input_path, output_path)
        print(f"Duplicates removed. Cleaned file saved as {output_path}")
    except FileNotFoundError:
        print(f"Error: File '{input_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")