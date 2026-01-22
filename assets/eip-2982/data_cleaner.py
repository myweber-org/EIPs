import re
from typing import List, Optional

def remove_special_characters(text: str, keep_spaces: bool = True) -> str:
    """
    Remove all non-alphanumeric characters from the input string.
    
    Args:
        text: The input string to clean.
        keep_spaces: If True, spaces are preserved. If False, spaces are removed.
    
    Returns:
        The cleaned string containing only alphanumeric characters and optionally spaces.
    """
    if keep_spaces:
        pattern = r'[^A-Za-z0-9\s]+'
    else:
        pattern = r'[^A-Za-z0-9]+'
    
    return re.sub(pattern, '', text)

def normalize_whitespace(text: str) -> str:
    """
    Replace multiple consecutive whitespace characters with a single space.
    
    Args:
        text: The input string to normalize.
    
    Returns:
        The string with normalized whitespace.
    """
    return re.sub(r'\s+', ' ', text).strip()

def clean_text_pipeline(text: str, 
                       remove_special: bool = True,
                       normalize_ws: bool = True,
                       to_lower: bool = False) -> str:
    """
    Apply a series of cleaning operations to the input text.
    
    Args:
        text: The input string to clean.
        remove_special: If True, remove special characters.
        normalize_ws: If True, normalize whitespace.
        to_lower: If True, convert text to lowercase.
    
    Returns:
        The cleaned text after applying all specified operations.
    """
    if not isinstance(text, str):
        return ''
    
    cleaned = text
    
    if remove_special:
        cleaned = remove_special_characters(cleaned)
    
    if normalize_ws:
        cleaned = normalize_whitespace(cleaned)
    
    if to_lower:
        cleaned = cleaned.lower()
    
    return cleaned

def batch_clean_texts(texts: List[str], **kwargs) -> List[str]:
    """
    Apply cleaning pipeline to a list of text strings.
    
    Args:
        texts: List of text strings to clean.
        **kwargs: Additional arguments to pass to clean_text_pipeline.
    
    Returns:
        List of cleaned text strings.
    """
    return [clean_text_pipeline(text, **kwargs) for text in texts]

def validate_and_clean(text: Optional[str], default: str = "") -> str:
    """
    Validate input and return cleaned text or default value.
    
    Args:
        text: The input text to validate and clean.
        default: The default value to return if text is invalid.
    
    Returns:
        Cleaned text or default value.
    """
    if text is None or not isinstance(text, str):
        return default
    
    cleaned = clean_text_pipeline(text)
    return cleaned if cleaned else default