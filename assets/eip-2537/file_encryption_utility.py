import os
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class FileEncryptor:
    def __init__(self, iterations=100000):
        self.iterations = iterations
        self.backend = default_backend()
    
    def derive_key(self, password, salt):
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.iterations,
            backend=self.backend
        )
        return kdf.derive(password.encode())
    
    def encrypt_file(self, input_path, output_path, password):
        salt = os.urandom(16)
        key = self.derive_key(password, salt)
        
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        
        padding_length = 16 - (len(plaintext) % 16)
        plaintext += bytes([padding_length]) * padding_length
        
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        with open(output_path, 'wb') as f:
            f.write(salt + iv + ciphertext)
        
        return True
    
    def decrypt_file(self, input_path, output_path, password):
        with open(input_path, 'rb') as f:
            data = f.read()
        
        salt = data[:16]
        iv = data[16:32]
        ciphertext = data[32:]
        
        key = self.derive_key(password, salt)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        decryptor = cipher.decryptor()
        
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        padding_length = plaintext[-1]
        plaintext = plaintext[:-padding_length]
        
        with open(output_path, 'wb') as f:
            f.write(plaintext)
        
        return True

def create_test_file():
    test_content = b"This is a test file for encryption demonstration.\n" * 10
    with open('test_plain.txt', 'wb') as f:
        f.write(test_content)
    print("Test file created: test_plain.txt")

if __name__ == "__main__":
    encryptor = FileEncryptor()
    
    create_test_file()
    
    password = "SecurePass123!"
    
    print("Encrypting file...")
    encryptor.encrypt_file('test_plain.txt', 'test_encrypted.bin', password)
    
    print("Decrypting file...")
    encryptor.decrypt_file('test_encrypted.bin', 'test_decrypted.txt', password)
    
    with open('test_plain.txt', 'rb') as f1, open('test_decrypted.txt', 'rb') as f2:
        if f1.read() == f2.read():
            print("Encryption/decryption successful!")
        else:
            print("Error: Decrypted content doesn't match original!")
    
    for temp_file in ['test_plain.txt', 'test_encrypted.bin', 'test_decrypted.txt']:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"Cleaned up: {temp_file}")