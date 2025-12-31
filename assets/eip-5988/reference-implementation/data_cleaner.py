import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: list of columns to apply strategy to
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = df.columns
    
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns:
            if strategy == 'mean':
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif strategy == 'median':
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif strategy == 'mode':
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_copy = df_copy.dropna(subset=[col])
    
    return df_copy

def normalize_columns(df, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to normalize
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns and df_copy[col].dtype in [np.float64, np.int64]:
            min_val = df_copy[col].min()
            max_val = df_copy[col].max()
            
            if max_val > min_val:
                df_copy[col] = (df_copy[col] - min_val) / (max_val - min_val)
    
    return df_copy

def clean_dataframe(df, operations=None):
    """
    Apply multiple cleaning operations to DataFrame.
    
    Args:
        df: pandas DataFrame
        operations: list of tuples (operation_name, kwargs)
    
    Returns:
        Cleaned DataFrame
    """
    if operations is None:
        operations = [
            ('remove_duplicates', {}),
            ('handle_missing_values', {'strategy': 'mean'}),
            ('normalize_columns', {})
        ]
    
    cleaned_df = df.copy()
    
    for operation, kwargs in operations:
        if operation == 'remove_duplicates':
            cleaned_df = remove_duplicates(cleaned_df, **kwargs)
        elif operation == 'handle_missing_values':
            cleaned_df = handle_missing_values(cleaned_df, **kwargs)
        elif operation == 'normalize_columns':
            cleaned_df = normalize_columns(cleaned_df, **kwargs)
    
    return cleaned_df