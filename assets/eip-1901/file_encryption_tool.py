import os
import sys

def xor_encrypt_decrypt(data, key):
    return bytes([b ^ key[i % len(key)] for i, b in enumerate(data)])

def process_file(input_path, output_path, key):
    try:
        with open(input_path, 'rb') as f:
            data = f.read()
        
        processed_data = xor_encrypt_decrypt(data, key)
        
        with open(output_path, 'wb') as f:
            f.write(processed_data)
        
        print(f"File processed successfully: {output_path}")
        return True
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

def main():
    if len(sys.argv) < 4:
        print("Usage: python file_encryption_tool.py <input_file> <output_file> <key>")
        print("Example: python file_encryption_tool.py secret.txt encrypted.txt mykey123")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    key = sys.argv[3].encode()
    
    if not os.path.exists(input_file):
        print(f"Input file does not exist: {input_file}")
        sys.exit(1)
    
    if process_file(input_file, output_file, key):
        print("Operation completed.")
    else:
        print("Operation failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()