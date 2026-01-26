import os
import sys

def xor_encrypt_decrypt(data, key):
    """Encrypt or decrypt data using XOR cipher."""
    key_length = len(key)
    return bytes([data[i] ^ key[i % key_length] for i in range(len(data))])

def process_file(input_file, output_file, key):
    """Read input file, process with XOR, and write to output file."""
    try:
        with open(input_file, 'rb') as f:
            file_data = f.read()
        
        processed_data = xor_encrypt_decrypt(file_data, key)
        
        with open(output_file, 'wb') as f:
            f.write(processed_data)
        
        print(f"Operation completed. Output saved to: {output_file}")
        return True
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    if len(sys.argv) != 4:
        print("Usage: python file_encryption_tool.py <input_file> <output_file> <key>")
        print("Example: python file_encryption_tool.py secret.txt encrypted.txt mykey123")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    key = sys.argv[3].encode('utf-8')
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)
    
    process_file(input_file, output_file, key)

if __name__ == "__main__":
    main()