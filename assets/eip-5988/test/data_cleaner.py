
import re

def clean_string(text):
    """
    Cleans a string by:
    1. Removing leading and trailing whitespace.
    2. Replacing multiple spaces/newlines/tabs with a single space.
    3. Converting the string to lowercase.
    
    Args:
        text (str): The input string to be cleaned.
    
    Returns:
        str: The cleaned and normalized string.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Replace multiple whitespace characters (spaces, newlines, tabs) with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text