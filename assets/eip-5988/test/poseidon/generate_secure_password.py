
import secrets
import string

def generate_password(length=16, use_uppercase=True, use_lowercase=True, use_digits=True, use_symbols=True):
    """
    Generate a cryptographically secure random password.
    
    Args:
        length: Length of the password (default: 16)
        use_uppercase: Include uppercase letters (default: True)
        use_lowercase: Include lowercase letters (default: True)
        use_digits: Include digits (default: True)
        use_symbols: Include symbols (default: True)
    
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
    if use_symbols:
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
        Tuple of (is_valid, message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_symbol = any(c in string.punctuation for c in password)
    
    criteria_met = sum([has_upper, has_lower, has_digit, has_symbol])
    
    if criteria_met >= 3:
        return True, "Strong password"
    elif criteria_met == 2:
        return True, "Moderate password"
    else:
        return False, "Weak password - include more character types"

if __name__ == "__main__":
    try:
        password = generate_password(
            length=20,
            use_uppercase=True,
            use_lowercase=True,
            use_digits=True,
            use_symbols=True
        )
        
        print(f"Generated Password: {password}")
        
        is_valid, message = validate_password_strength(password)
        print(f"Strength Check: {message}")
        
    except ValueError as e:
        print(f"Error: {e}")