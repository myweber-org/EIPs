import os
import sys

def xor_cipher(data, key):
    """Encrypt or decrypt data using XOR cipher."""
    return bytes([b ^ key for b in data])

def process_file(input_path, output_path, key):
    """Read a file, encrypt/decrypt it, and write to output."""
    try:
        with open(input_path, 'rb') as f:
            data = f.read()
        
        processed_data = xor_cipher(data, key)
        
        with open(output_path, 'wb') as f:
            f.write(processed_data)
        
        print(f"Successfully processed {input_path} -> {output_path}")
        return True
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

def main():
    if len(sys.argv) != 4:
        print("Usage: python file_encryptor.py <input_file> <output_file> <key>")
        print("Key must be an integer between 0 and 255")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        key = int(sys.argv[3])
        if not 0 <= key <= 255:
            raise ValueError
    except ValueError:
        print("Key must be an integer between 0 and 255")
        sys.exit(1)
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        sys.exit(1)
    
    if process_file(input_file, output_file, key):
        print("Operation completed successfully")
    else:
        print("Operation failed")

if __name__ == "__main__":
    main()