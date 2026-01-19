import pandas as pd
import numpy as np
from typing import Optional, List

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop', fill_value: Optional[float] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'drop' to remove rows, 'fill' to fill values
        fill_value: Value to fill when strategy is 'fill'
    
    Returns:
        DataFrame with handled missing values
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        if fill_value is not None:
            return df.fillna(fill_value)
        else:
            return df.fillna(df.mean(numeric_only=True))
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")

def normalize_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Normalize specified columns to range [0, 1].
    
    Args:
        df: Input DataFrame
        columns: List of column names to normalize
    
    Returns:
        DataFrame with normalized columns
    """
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns and df_copy[col].dtype in [np.float64, np.int64]:
            col_min = df_copy[col].min()
            col_max = df_copy[col].max()
            if col_max != col_min:
                df_copy[col] = (df_copy[col] - col_min) / (col_max - col_min)
    return df_copy

def clean_dataframe(df: pd.DataFrame, 
                    remove_dups: bool = True,
                    handle_nulls: bool = True,
                    normalize_numeric: bool = False) -> pd.DataFrame:
    """
    Main cleaning pipeline for DataFrame.
    
    Args:
        df: Input DataFrame
        remove_dups: Whether to remove duplicates
        handle_nulls: Whether to handle missing values
        normalize_numeric: Whether to normalize numeric columns
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if handle_nulls:
        cleaned_df = handle_missing_values(cleaned_df, strategy='fill')
    
    if normalize_numeric:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            cleaned_df = normalize_columns(cleaned_df, numeric_cols)
    
    return cleaned_df