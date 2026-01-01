import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_column(df, column_name, fill_method='mean'):
    """
    Clean a numeric column by handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of the column to clean.
        fill_method (str): Method to fill missing values.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    if fill_method == 'mean':
        fill_value = df[column_name].mean()
    elif fill_method == 'median':
        fill_value = df[column_name].median()
    elif fill_method == 'zero':
        fill_value = 0
    else:
        raise ValueError("Invalid fill_method. Choose from 'mean', 'median', or 'zero'")
    
    df_cleaned = df.copy()
    df_cleaned[column_name] = df_cleaned[column_name].fillna(fill_value)
    
    return df_cleaned

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"import re
import pandas as pd
from typing import Optional, List, Dict, Any

def clean_string(text: str) -> str:
    """
    Clean a string by removing extra whitespace and converting to lowercase.
    """
    if not isinstance(text, str):
        return ''
    cleaned = re.sub(r'\s+', ' ', text.strip())
    return cleaned.lower()

def validate_email(email: str) -> bool:
    """
    Validate an email address format.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def remove_duplicates(data: List[Any]) -> List[Any]:
    """
    Remove duplicates from a list while preserving order.
    """
    seen = set()
    result = []
    for item in data:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame column names to lowercase with underscores.
    """
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    return df

def filter_by_threshold(data: List[float], threshold: float) -> List[float]:
    """
    Filter numeric values above a given threshold.
    """
    return [value for value in data if isinstance(value, (int, float)) and value > threshold]

def count_unique_items(items: List[Any]) -> Dict[Any, int]:
    """
    Count occurrences of unique items in a list.
    """
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts