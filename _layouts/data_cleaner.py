
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df: pandas DataFrame to clean
        column_mapping: dict mapping old column names to new ones
        drop_duplicates: bool, whether to remove duplicate rows
        normalize_text: bool, whether to normalize text columns
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if normalize_text:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            cleaned_df[col] = cleaned_df[col].apply(_normalize_string)
    
    return cleaned_df

def _normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    """
    if pd.isna(text):
        return text
    
    normalized = str(text).lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = re.sub(r'[^\w\s-]', '', normalized)
    
    return normalized

def validate_dataframe(df, required_columns=None, allow_nulls=True):
    """
    Validate a DataFrame for required columns and null values.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of column names that must be present
        allow_nulls: bool, whether null values are allowed
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            return False, f"Missing required columns: {missing}"
    
    if not allow_nulls and df.isnull().any().any():
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0].index.tolist()
        return False, f"Null values found in columns: {null_cols}"
    
    return True, "DataFrame validation passed"