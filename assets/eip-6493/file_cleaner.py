
import sys

def remove_empty_lines(input_file, output_file=None):
    """
    Reads a text file, removes all empty lines, and writes the result
    to an output file. If no output file is specified, overwrites the input file.
    """
    if output_file is None:
        output_file = input_file

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Filter out lines that are empty or contain only whitespace
        non_empty_lines = [line for line in lines if line.strip()]

        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(non_empty_lines)

        print(f"Removed {len(lines) - len(non_empty_lines)} empty lines.")
        print(f"Output written to: {output_file}")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_cleaner.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    remove_empty_lines(input_file, output_file)