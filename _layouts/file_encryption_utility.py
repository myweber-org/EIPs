import os
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

def derive_key(password, salt):
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return kdf.derive(password.encode())

def encrypt_file(input_file, output_file, password):
    salt = os.urandom(16)
    key = derive_key(password, salt)
    iv = os.urandom(16)
    
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    
    padder = padding.PKCS7(128).padder()
    
    with open(input_file, 'rb') as f:
        plaintext = f.read()
    
    padded_data = padder.update(plaintext) + padder.finalize()
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    
    with open(output_file, 'wb') as f:
        f.write(salt + iv + ciphertext)

def decrypt_file(input_file, output_file, password):
    with open(input_file, 'rb') as f:
        data = f.read()
    
    salt = data[:16]
    iv = data[16:32]
    ciphertext = data[32:]
    
    key = derive_key(password, salt)
    
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    
    padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    
    unpadder = padding.PKCS7(128).unpadder()
    plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
    
    with open(output_file, 'wb') as f:
        f.write(plaintext)

def main():
    action = input("Enter 'e' to encrypt or 'd' to decrypt: ").strip().lower()
    input_file = input("Enter input file path: ").strip()
    output_file = input("Enter output file path: ").strip()
    password = input("Enter password: ").strip()
    
    if action == 'e':
        encrypt_file(input_file, output_file, password)
        print("File encrypted successfully.")
    elif action == 'd':
        decrypt_file(input_file, output_file, password)
        print("File decrypted successfully.")
    else:
        print("Invalid action.")

if __name__ == "__main__":
    main()