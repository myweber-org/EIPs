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

def process_file(input_path: str, output_path: str, key: str, mode: str = 'encrypt'):
    cipher = XORCipher(key)
    
    try:
        with open(input_path, 'rb') as f:
            data = f.read()
        
        if mode == 'encrypt':
            processed_data = cipher.encrypt(data)
        elif mode == 'decrypt':
            processed_data = cipher.decrypt(data)
        else:
            raise ValueError("Mode must be 'encrypt' or 'decrypt'")
        
        with open(output_path, 'wb') as f:
            f.write(processed_data)
        
        print(f"File {mode}ed successfully: {output_path}")
        
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}")
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    if len(sys.argv) < 5:
        print("Usage: python file_encryptor.py <input_file> <output_file> <key> <mode>")
        print("Modes: encrypt, decrypt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    key = sys.argv[3]
    mode = sys.argv[4].lower()
    
    if mode not in ['encrypt', 'decrypt']:
        print("Error: Mode must be 'encrypt' or 'decrypt'")
        sys.exit(1)
    
    process_file(input_file, output_file, key, mode)

if __name__ == "__main__":
    main()