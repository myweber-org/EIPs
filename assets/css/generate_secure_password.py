import secrets
import string

def generate_password(length=16, use_uppercase=True, use_lowercase=True, 
                     use_digits=True, use_special=True):
    """
    Generate a secure random password with customizable character sets.
    
    Args:
        length: Length of the password (default: 16)
        use_uppercase: Include uppercase letters (default: True)
        use_lowercase: Include lowercase letters (default: True)
        use_digits: Include digits (default: True)
        use_special: Include special characters (default: True)
    
    Returns:
        A secure random password string
    
    Raises:
        ValueError: If no character sets are selected
    """
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
    
    if length < 8:
        raise ValueError("Password length must be at least 8 characters")
    
    password = ''.join(secrets.choice(character_pool) for _ in range(length))
    
    return password

def validate_password_strength(password):
    """
    Validate password strength based on common security criteria.
    
    Args:
        password: Password string to validate
    
    Returns:
        Dictionary with validation results
    """
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in string.punctuation for c in password)
    
    return {
        'length_ok': len(password) >= 12,
        'has_uppercase': has_upper,
        'has_lowercase': has_lower,
        'has_digit': has_digit,
        'has_special': has_special,
        'is_strong': len(password) >= 12 and has_upper and has_lower and has_digit and has_special
    }

if __name__ == "__main__":
    try:
        password = generate_password(length=20)
        print(f"Generated password: {password}")
        
        strength = validate_password_strength(password)
        print(f"Password strength analysis:")
        for key, value in strength.items():
            print(f"  {key}: {value}")
        
        if strength['is_strong']:
            print("Password is strong!")
        else:
            print("Password could be stronger. Consider increasing length or adding more character types.")
            
    except ValueError as e:
        print(f"Error: {e}")