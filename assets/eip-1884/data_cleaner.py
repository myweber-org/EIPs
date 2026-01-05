import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """Remove duplicate rows from DataFrame."""
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """Fill missing values using specified strategy."""
    if columns is None:
        columns = df.columns
    
    df_filled = df.copy()
    for col in columns:
        if df[col].dtype in [np.float64, np.int64]:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            else:
                fill_value = 0
            df_filled[col].fillna(fill_value, inplace=True)
        else:
            df_filled[col].fillna('', inplace=True)
    return df_filled

def normalize_column(df, column):
    """Normalize numeric column to 0-1 range."""
    if df[column].dtype in [np.float64, np.int64]:
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val > min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataframe(df, operations=['remove_duplicates', 'fill_missing']):
    """Apply multiple cleaning operations to DataFrame."""
    cleaned_df = df.copy()
    
    if 'remove_duplicates' in operations:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if 'fill_missing' in operations:
        cleaned_df = fill_missing_values(cleaned_df)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """Validate DataFrame structure and content."""
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return True