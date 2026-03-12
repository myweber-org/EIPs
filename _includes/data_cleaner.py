
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

def fill_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fill missing values in DataFrame columns.
    
    Args:
        df: Input DataFrame
        strategy: Method to fill missing values ('mean', 'median', 'mode', 'zero')
        columns: Specific columns to fill, fills all columns if None
    
    Returns:
        DataFrame with missing values filled
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
            elif strategy == 'zero':
                fill_value = 0
            else:
                fill_value = df[col].mean()
            
            df_filled[col] = df[col].fillna(fill_value)
        else:
            # For non-numeric columns, fill with most frequent value
            if not df[col].mode().empty:
                df_filled[col] = df[col].fillna(df[col].mode()[0])
    
    return df_filled

def normalize_columns(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Normalize numeric columns to range [0, 1].
    
    Args:
        df: Input DataFrame
        columns: Columns to normalize, normalizes all numeric columns if None
    
    Returns:
        DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_max != col_min:
                df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                df_normalized[col] = 0
    
    return df_normalized

def remove_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None, threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove outliers using z-score method.
    
    Args:
        df: Input DataFrame
        columns: Columns to check for outliers
        threshold: Z-score threshold for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    mask = pd.Series([True] * len(df))
    
    for col in columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            mask = mask & (z_scores < threshold)
    
    return df[mask].reset_index(drop=True)

def clean_dataframe(df: pd.DataFrame, 
                    remove_dups: bool = True,
                    fill_missing: bool = True,
                    normalize: bool = False,
                    remove_out: bool = False) -> pd.DataFrame:
    """
    Apply multiple cleaning operations to DataFrame.
    
    Args:
        df: Input DataFrame
        remove_dups: Whether to remove duplicates
        fill_missing: Whether to fill missing values
        normalize: Whether to normalize numeric columns
        remove_out: Whether to remove outliers
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if fill_missing:
        cleaned_df = fill_missing_values(cleaned_df)
    
    if normalize:
        cleaned_df = normalize_columns(cleaned_df)
    
    if remove_out:
        cleaned_df = remove_outliers(cleaned_df)
    
    return cleaned_df