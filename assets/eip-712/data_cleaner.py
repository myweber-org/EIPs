
import re

def clean_string(text):
    """
    Clean a string by removing extra whitespace and converting to lowercase.
    
    Args:
        text (str): The input string to clean.
    
    Returns:
        str: The cleaned string.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text

def normalize_names(name_list):
    """
    Normalize a list of names by cleaning each string.
    
    Args:
        name_list (list): List of name strings.
    
    Returns:
        list: List of cleaned names.
    """
    if not isinstance(name_list, list):
        raise TypeError("Input must be a list")
    
    return [clean_string(name) for name in name_list]