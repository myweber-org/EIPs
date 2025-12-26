
import os
import sys

def xor_cipher(data, key):
    """XOR encrypt/decrypt data with key."""
    return bytes([b ^ key for b in data])

def encrypt_file(input_path, output_path, key):
    """Encrypt a file using XOR cipher."""
    try:
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        ciphertext = xor_cipher(plaintext, key)
        with open(output_path, 'wb') as f:
            f.write(ciphertext)
        print(f"Encrypted: {input_path} -> {output_path}")
        return True
    except Exception as e:
        print(f"Encryption failed: {e}")
        return False

def decrypt_file(input_path, output_path, key):
    """Decrypt a file using XOR cipher."""
    return encrypt_file(input_path, output_path, key)

def main():
    if len(sys.argv) < 5:
        print("Usage: python file_encryption_tool.py <encrypt|decrypt> <input_file> <output_file> <key>")
        print("Key must be an integer between 0 and 255")
        sys.exit(1)

    operation = sys.argv[1].lower()
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    try:
        key = int(sys.argv[4])
        if not 0 <= key <= 255:
            raise ValueError
    except ValueError:
        print("Error: Key must be an integer between 0 and 255")
        sys.exit(1)

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)

    if operation == 'encrypt':
        encrypt_file(input_file, output_file, key)
    elif operation == 'decrypt':
        decrypt_file(input_file, output_file, key)
    else:
        print("Error: Operation must be 'encrypt' or 'decrypt'")
        sys.exit(1)

if __name__ == "__main__":
    main()