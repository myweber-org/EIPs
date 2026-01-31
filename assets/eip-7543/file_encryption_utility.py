
import os
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class FileEncryptor:
    def __init__(self, password: str, salt_length: int = 16):
        self.password = password.encode()
        self.salt_length = salt_length

    def derive_key(self, salt: bytes, iterations: int = 100000) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
            backend=default_backend()
        )
        return kdf.derive(self.password)

    def encrypt_file(self, input_path: str, output_path: str) -> bool:
        try:
            with open(input_path, 'rb') as f:
                plaintext = f.read()

            salt = os.urandom(self.salt_length)
            key = self.derive_key(salt)
            iv = os.urandom(16)

            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()

            # Pad plaintext to AES block size
            pad_length = 16 - (len(plaintext) % 16)
            plaintext += bytes([pad_length] * pad_length)

            ciphertext = encryptor.update(plaintext) + encryptor.finalize()

            with open(output_path, 'wb') as f:
                f.write(salt + iv + ciphertext)

            return True
        except Exception:
            return False

    def decrypt_file(self, input_path: str, output_path: str) -> bool:
        try:
            with open(input_path, 'rb') as f:
                data = f.read()

            salt = data[:self.salt_length]
            iv = data[self.salt_length:self.salt_length + 16]
            ciphertext = data[self.salt_length + 16:]

            key = self.derive_key(salt)

            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()

            plaintext = decryptor.update(ciphertext) + decryptor.finalize()

            # Remove padding
            pad_length = plaintext[-1]
            plaintext = plaintext[:-pad_length]

            with open(output_path, 'wb') as f:
                f.write(plaintext)

            return True
        except Exception:
            return False

def main():
    encryptor = FileEncryptor("secure_password_123")
    
    # Example usage
    test_data = b"This is a secret message that needs encryption."
    with open("test_plain.txt", "wb") as f:
        f.write(test_data)
    
    # Encrypt
    if encryptor.encrypt_file("test_plain.txt", "test_encrypted.bin"):
        print("Encryption successful")
    
    # Decrypt
    if encryptor.decrypt_file("test_encrypted.bin", "test_decrypted.txt"):
        print("Decryption successful")
    
    # Cleanup
    for fname in ["test_plain.txt", "test_encrypted.bin", "test_decrypted.txt"]:
        if os.path.exists(fname):
            os.remove(fname)

if __name__ == "__main__":
    main()