import pandas as pd
import numpy as np
from typing import List, Optional

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase with underscores."""
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def remove_duplicate_rows(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """Remove duplicate rows based on specified columns or all columns."""
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Fill missing values using specified strategy."""
    df_filled = df.copy()
    
    if columns is None:
        columns = df.columns.tolist()
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
            elif strategy == 'median':
                df_filled[col] = df_filled[col].fillna(df_filled[col].median())
            elif strategy == 'zero':
                df_filled[col] = df_filled[col].fillna(0)
        else:
            df_filled[col] = df_filled[col].fillna('unknown')
    
    return df_filled

def standardize_text_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Standardize text columns to lowercase and strip whitespace."""
    df_std = df.copy()
    for col in columns:
        if col in df_std.columns:
            df_std[col] = df_std[col].astype(str).str.lower().str.strip()
    return df_std

def clean_dataset(df: pd.DataFrame, text_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Apply all cleaning functions to dataset."""
    df_clean = df.copy()
    df_clean = clean_column_names(df_clean)
    df_clean = remove_duplicate_rows(df_clean)
    
    if text_columns:
        df_clean = standardize_text_columns(df_clean, text_columns)
    
    df_clean = fill_missing_values(df_clean)
    return df_clean

if __name__ == "__main__":
    sample_data = {
        'Product Name': ['Laptop', 'Mouse', 'Laptop', 'Keyboard', None],
        'Price': [999.99, 25.50, 999.99, 45.75, 30.00],
        'Category': ['Electronics', 'Electronics', 'electronics', 'Electronics', 'Accessories']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataset(df, text_columns=['Product Name', 'Category'])
    print(cleaned_df)