
import os
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class FileEncryptor:
    def __init__(self, password: str, salt: bytes = None):
        self.password = password.encode()
        self.salt = salt or os.urandom(16)
        self.backend = default_backend()
        
    def _derive_key(self, length: int = 32) -> bytes:
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=length,
            salt=self.salt,
            iterations=100000,
            backend=self.backend
        )
        return kdf.derive(self.password)
    
    def encrypt_file(self, input_path: str, output_path: str) -> dict:
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        
        key = self._derive_key()
        iv = os.urandom(16)
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        
        pad_length = 16 - (len(plaintext) % 16)
        padded_data = plaintext + bytes([pad_length] * pad_length)
        
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        with open(output_path, 'wb') as f:
            f.write(self.salt + iv + ciphertext)
        
        return {
            'original_size': len(plaintext),
            'encrypted_size': len(ciphertext),
            'salt': base64.b64encode(self.salt).decode(),
            'iv': base64.b64encode(iv).decode()
        }
    
    def decrypt_file(self, input_path: str, output_path: str) -> dict:
        with open(input_path, 'rb') as f:
            data = f.read()
        
        self.salt = data[:16]
        iv = data[16:32]
        ciphertext = data[32:]
        
        key = self._derive_key()
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        
        padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        pad_length = padded_plaintext[-1]
        plaintext = padded_plaintext[:-pad_length]
        
        with open(output_path, 'wb') as f:
            f.write(plaintext)
        
        return {
            'decrypted_size': len(plaintext),
            'padding_removed': pad_length
        }

def create_test_file(path: str, size_kb: int = 1) -> None:
    test_data = os.urandom(size_kb * 1024)
    with open(path, 'wb') as f:
        f.write(test_data)

if __name__ == "__main__":
    test_file = "test_data.bin"
    encrypted_file = "encrypted.bin"
    decrypted_file = "decrypted.bin"
    
    create_test_file(test_file, 5)
    
    encryptor = FileEncryptor("secure_password_123")
    
    enc_result = encryptor.encrypt_file(test_file, encrypted_file)
    print(f"Encryption complete: {enc_result}")
    
    decryptor = FileEncryptor("secure_password_123", base64.b64decode(enc_result['salt']))
    dec_result = decryptor.decrypt_file(encrypted_file, decrypted_file)
    print(f"Decryption complete: {dec_result}")
    
    with open(test_file, 'rb') as f1, open(decrypted_file, 'rb') as f2:
        if f1.read() == f2.read():
            print("File verification: SUCCESS")
        else:
            print("File verification: FAILED")
    
    for f in [test_file, encrypted_file, decrypted_file]:
        if os.path.exists(f):
            os.remove(f)