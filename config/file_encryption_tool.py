
import os
import sys

class XORCipher:
    def __init__(self, key: str):
        self.key = key.encode('utf-8')
    
    def encrypt(self, data: bytes) -> bytes:
        encrypted = bytearray()
        key_length = len(self.key)
        for i, byte in enumerate(data):
            encrypted.append(byte ^ self.key[i % key_length])
        return bytes(encrypted)
    
    def decrypt(self, data: bytes) -> bytes:
        return self.encrypt(data)

def process_file(input_path: str, output_path: str, key: str, mode: str):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return False
    
    cipher = XORCipher(key)
    
    try:
        with open(input_path, 'rb') as f:
            data = f.read()
        
        if mode == 'encrypt':
            processed_data = cipher.encrypt(data)
            action = "Encrypted"
        elif mode == 'decrypt':
            processed_data = cipher.decrypt(data)
            action = "Decrypted"
        else:
            print("Error: Mode must be 'encrypt' or 'decrypt'.")
            return False
        
        with open(output_path, 'wb') as f:
            f.write(processed_data)
        
        print(f"{action} file saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

def main():
    if len(sys.argv) != 5:
        print("Usage: python file_encryption_tool.py <input_file> <output_file> <key> <encrypt|decrypt>")
        print("Example: python file_encryption_tool.py secret.txt secret.enc mypassword encrypt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    key = sys.argv[3]
    mode = sys.argv[4].lower()
    
    if mode not in ['encrypt', 'decrypt']:
        print("Error: Mode must be 'encrypt' or 'decrypt'.")
        sys.exit(1)
    
    success = process_file(input_file, output_file, key, mode)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()