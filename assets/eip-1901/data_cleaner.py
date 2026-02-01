
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a DataFrame by removing duplicates and normalizing string columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns_to_clean (list, optional): List of column names to normalize.
            If None, all object dtype columns are cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # Remove duplicate rows
    df_clean = df_clean.drop_duplicates()
    
    # Determine columns to normalize
    if columns_to_clean is None:
        columns_to_clean = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    # Normalize string columns
    for col in columns_to_clean:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].apply(_normalize_string)
    
    return df_clean

def _normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    
    Args:
        text (str): Input string.
    
    Returns:
        str: Normalized string.
    """
    if pd.isna(text):
        return text
    
    text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove special characters except alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    return text

def validate_email_column(df, email_column):
    """
    Validate email addresses in a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of the column containing email addresses.
    
    Returns:
        pd.DataFrame: DataFrame with validation results.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    df_result = df.copy()
    df_result['is_valid_email'] = df_result[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notna(x) else False
    )
    
    return df_result