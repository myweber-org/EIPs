import os
import sys

def xor_cipher(data, key):
    """Encrypt or decrypt data using XOR with the given key."""
    key_length = len(key)
    return bytes([data[i] ^ key[i % key_length] for i in range(len(data))])

def process_file(input_path, output_path, key):
    """Read input file, encrypt/decrypt, and write to output file."""
    try:
        with open(input_path, 'rb') as f:
            file_data = f.read()
        
        processed_data = xor_cipher(file_data, key)
        
        with open(output_path, 'wb') as f:
            f.write(processed_data)
        
        print(f"Successfully processed '{input_path}' -> '{output_path}'")
        return True
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

def main():
    if len(sys.argv) != 4:
        print("Usage: python file_encryption_tool.py <input_file> <output_file> <key>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    key = sys.argv[3].encode('utf-8')
    
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' does not exist.")
        sys.exit(1)
    
    if os.path.exists(output_file):
        overwrite = input(f"Output file '{output_file}' exists. Overwrite? (y/n): ")
        if overwrite.lower() != 'y':
            print("Operation cancelled.")
            sys.exit(0)
    
    success = process_file(input_file, output_file, key)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()