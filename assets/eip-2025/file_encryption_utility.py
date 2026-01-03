import os
import sys

def xor_cipher(data, key):
    """XOR cipher for encryption/decryption."""
    key_length = len(key)
    return bytes([data[i] ^ key[i % key_length] for i in range(len(data))])

def encrypt_file(input_path, output_path, key):
    """Encrypt a file using XOR cipher."""
    try:
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        ciphertext = xor_cipher(plaintext, key.encode())
        with open(output_path, 'wb') as f:
            f.write(ciphertext)
        print(f"File encrypted successfully: {output_path}")
    except Exception as e:
        print(f"Error encrypting file: {e}")

def decrypt_file(input_path, output_path, key):
    """Decrypt a file using XOR cipher."""
    encrypt_file(input_path, output_path, key)  # XOR is symmetric

def main():
    if len(sys.argv) < 5:
        print("Usage: python file_encryption_utility.py <encrypt|decrypt> <input_file> <output_file> <key>")
        sys.exit(1)

    mode = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    key = sys.argv[4]

    if not os.path.exists(input_file):
        print(f"Input file does not exist: {input_file}")
        sys.exit(1)

    if mode == 'encrypt':
        encrypt_file(input_file, output_file, key)
    elif mode == 'decrypt':
        decrypt_file(input_file, output_file, key)
    else:
        print("Invalid mode. Use 'encrypt' or 'decrypt'.")

if __name__ == "__main__":
    main()