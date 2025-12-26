
import pandas as pd
import numpy as np
from typing import List, Union

def remove_duplicates(dataframe: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        dataframe: Input DataFrame
        subset: List of column names to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return dataframe.drop_duplicates(subset=subset, keep='first')

def normalize_column(dataframe: pd.DataFrame, column: str, method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize values in a specified column.
    
    Args:
        dataframe: Input DataFrame
        column: Name of column to normalize
        method: Normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized column
    """
    df = dataframe.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val > min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val > 0:
            df[column] = (df[column] - mean_val) / std_val
    
    return df

def handle_missing_values(dataframe: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in numeric columns.
    
    Args:
        dataframe: Input DataFrame
        strategy: Imputation strategy ('mean', 'median', or 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    df = dataframe.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        df = df.dropna(subset=numeric_cols)
    elif strategy == 'mean':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
    elif strategy == 'median':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    
    return df

def clean_dataset(dataframe: pd.DataFrame, 
                  deduplicate: bool = True,
                  normalize_columns: List[str] = None,
                  missing_strategy: str = 'mean') -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: Input DataFrame
        deduplicate: Whether to remove duplicates
        normalize_columns: List of columns to normalize
        missing_strategy: Strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    df = dataframe.copy()
    
    if deduplicate:
        df = remove_duplicates(df)
    
    df = handle_missing_values(df, strategy=missing_strategy)
    
    if normalize_columns:
        for col in normalize_columns:
            if col in df.columns:
                df = normalize_column(df, col)
    
    return df