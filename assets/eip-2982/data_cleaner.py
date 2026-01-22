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
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: If True, remove duplicate rows
        fill_missing: Strategy for filling missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        Cleaned pandas DataFrame
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    if fill_missing == 'drop':
        df_clean = df_clean.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    elif fill_missing == 'mode':
        for col in df_clean.columns:
            mode_val = df_clean[col].mode()
            if not mode_val.empty:
                df_clean[col] = df_clean[col].fillna(mode_val.iloc[0])
    
    return df_clean

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
        min_rows: Minimum number of rows required
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    
    Args:
        df: DataFrame containing the data
        column: Column name to check for outliers
        method: Outlier detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_clean = df.copy()
    
    if method == 'iqr':
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
    
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df_clean[column].dropna()))
        df_clean = df_clean[z_scores < threshold]
    
    return df_clean