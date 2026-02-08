import os
import sys

class XORCipher:
    def __init__(self, key: str):
        self.key = key.encode('utf-8')
    
    def _crypt(self, data: bytes) -> bytes:
        key_length = len(self.key)
        return bytes([data[i] ^ self.key[i % key_length] for i in range(len(data))])
    
    def encrypt_file(self, input_path: str, output_path: str = None):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if output_path is None:
            output_path = input_path + '.enc'
        
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        
        ciphertext = self._crypt(plaintext)
        
        with open(output_path, 'wb') as f:
            f.write(ciphertext)
        
        return output_path
    
    def decrypt_file(self, input_path: str, output_path: str = None):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if output_path is None:
            if input_path.endswith('.enc'):
                output_path = input_path[:-4]
            else:
                output_path = input_path + '.dec'
        
        with open(input_path, 'rb') as f:
            ciphertext = f.read()
        
        plaintext = self._crypt(ciphertext)
        
        with open(output_path, 'wb') as f:
            f.write(plaintext)
        
        return output_path

def main():
    if len(sys.argv) < 4:
        print("Usage: python file_encryption_tool.py <encrypt|decrypt> <input_file> <key> [output_file]")
        sys.exit(1)
    
    operation = sys.argv[1].lower()
    input_file = sys.argv[2]
    key = sys.argv[3]
    output_file = sys.argv[4] if len(sys.argv) > 4 else None
    
    cipher = XORCipher(key)
    
    try:
        if operation == 'encrypt':
            result = cipher.encrypt_file(input_file, output_file)
            print(f"File encrypted successfully: {result}")
        elif operation == 'decrypt':
            result = cipher.decrypt_file(input_file, output_file)
            print(f"File decrypted successfully: {result}")
        else:
            print("Invalid operation. Use 'encrypt' or 'decrypt'.")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()