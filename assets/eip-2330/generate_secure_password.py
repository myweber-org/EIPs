import secrets
import string

def generate_password(length=16, use_uppercase=True, use_lowercase=True, use_digits=True, use_special=True):
    if length < 8:
        raise ValueError("Password length must be at least 8 characters")
    
    character_pool = ""
    
    if use_uppercase:
        character_pool += string.ascii_uppercase
    if use_lowercase:
        character_pool += string.ascii_lowercase
    if use_digits:
        character_pool += string.digits
    if use_special:
        character_pool += string.punctuation
    
    if not character_pool:
        raise ValueError("At least one character set must be selected")
    
    password = ''.join(secrets.choice(character_pool) for _ in range(length))
    
    return password

if __name__ == "__main__":
    try:
        password = generate_password(length=20, use_uppercase=True, use_lowercase=True, use_digits=True, use_special=True)
        print(f"Generated password: {password}")
    except ValueError as e:
        print(f"Error: {e}")