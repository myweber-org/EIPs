import os
import sys

class XORCipher:
    def __init__(self, key: str):
        self.key = key.encode('utf-8')
    
    def _crypt(self, data: bytes) -> bytes:
        key_length = len(self.key)
        return bytes([data[i] ^ self.key[i % key_length] for i in range(len(data))])
    
    def encrypt_file(self, input_path: str, output_path: str):
        try:
            with open(input_path, 'rb') as f:
                plaintext = f.read()
            ciphertext = self._crypt(plaintext)
            with open(output_path, 'wb') as f:
                f.write(ciphertext)
            return True
        except Exception as e:
            print(f"Encryption error: {e}")
            return False
    
    def decrypt_file(self, input_path: str, output_path: str):
        return self.encrypt_file(input_path, output_path)

def main():
    if len(sys.argv) < 4:
        print("Usage: python file_encryptor.py <encrypt|decrypt> <input_file> <output_file> [key]")
        sys.exit(1)
    
    operation = sys.argv[1].lower()
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    key = sys.argv[4] if len(sys.argv) > 4 else "default_secret_key"
    
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' not found.")
        sys.exit(1)
    
    cipher = XORCipher(key)
    
    if operation == 'encrypt':
        success = cipher.encrypt_file(input_file, output_file)
        if success:
            print(f"File encrypted successfully: {output_file}")
        else:
            print("Encryption failed.")
    elif operation == 'decrypt':
        success = cipher.decrypt_file(input_file, output_file)
        if success:
            print(f"File decrypted successfully: {output_file}")
        else:
            print("Decryption failed.")
    else:
        print("Invalid operation. Use 'encrypt' or 'decrypt'.")

if __name__ == "__main__":
    main()