
import os
import sys

class XORCipher:
    def __init__(self, key: str):
        self.key = key.encode('utf-8')
    
    def encrypt(self, data: bytes) -> bytes:
        key_length = len(self.key)
        return bytes([data[i] ^ self.key[i % key_length] for i in range(len(data))])
    
    def decrypt(self, data: bytes) -> bytes:
        return self.encrypt(data)

def process_file(input_path: str, output_path: str, key: str, mode: str):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)
    
    cipher = XORCipher(key)
    
    try:
        with open(input_path, 'rb') as f:
            file_data = f.read()
        
        if mode == 'encrypt':
            processed_data = cipher.encrypt(file_data)
            action = "Encrypted"
        elif mode == 'decrypt':
            processed_data = cipher.decrypt(file_data)
            action = "Decrypted"
        else:
            print("Error: Mode must be 'encrypt' or 'decrypt'.")
            sys.exit(1)
        
        with open(output_path, 'wb') as f:
            f.write(processed_data)
        
        print(f"{action} file saved to: {output_path}")
        print(f"Original size: {len(file_data)} bytes")
        print(f"Processed size: {len(processed_data)} bytes")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 5:
        print("Usage: python file_encryption_tool.py <input_file> <output_file> <key> <mode>")
        print("Modes: 'encrypt' or 'decrypt'")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    key = sys.argv[3]
    mode = sys.argv[4].lower()
    
    process_file(input_file, output_file, key, mode)

if __name__ == "__main__":
    main()