
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_mapping (dict, optional): Dictionary mapping old column names to new ones.
        drop_duplicates (bool): Whether to remove duplicate rows.
        normalize_text (bool): Whether to normalize text columns (strip, lower case).
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    if normalize_text:
        for col in cleaned_df.select_dtypes(include=['object']).columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
    
    return cleaned_df

def remove_special_characters(text, keep_pattern=r'[a-zA-Z0-9\s]'):
    """
    Remove special characters from a string, keeping only alphanumeric and spaces by default.
    
    Args:
        text (str): Input string.
        keep_pattern (str): Regex pattern of characters to keep.
    
    Returns:
        str: Cleaned string.
    """
    if not isinstance(text, str):
        return text
    
    return re.sub(f'[^{keep_pattern}]', '', text)

def validate_email(email):
    """
    Validate email format using regex.
    
    Args:
        email (str): Email address to validate.
    
    Returns:
        bool: True if email format is valid, False otherwise.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email)) if isinstance(email, str) else False