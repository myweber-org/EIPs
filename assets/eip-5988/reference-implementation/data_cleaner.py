import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalize a numeric column to range [0, 1].
    
    Args:
        df: Input DataFrame
        column: Name of column to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    col_min = df[column].min()
    col_max = df[column].max()
    
    if col_max == col_min:
        df[column] = 0.5
    else:
        df[column] = (df[column] - col_min) / (col_max - col_min)
    
    return df

def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in numeric columns.
    
    Args:
        df: Input DataFrame
        strategy: 'mean', 'median', or 'zero'
    
    Returns:
        DataFrame with missing values handled
    """
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            if strategy == 'mean':
                fill_value = df_clean[col].mean()
            elif strategy == 'median':
                fill_value = df_clean[col].median()
            elif strategy == 'zero':
                fill_value = 0
            else:
                raise ValueError("Strategy must be 'mean', 'median', or 'zero'")
            
            df_clean[col] = df_clean[col].fillna(fill_value)
    
    return df_clean

def clean_dataset(df: pd.DataFrame, 
                  deduplicate: bool = True,
                  normalize_cols: Optional[List[str]] = None,
                  missing_strategy: str = 'mean') -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        deduplicate: Whether to remove duplicates
        normalize_cols: Columns to normalize
        missing_strategy: Strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if deduplicate:
        df_clean = remove_duplicates(df_clean)
    
    df_clean = handle_missing_values(df_clean, strategy=missing_strategy)
    
    if normalize_cols:
        for col in normalize_cols:
            if col in df_clean.columns:
                df_clean = normalize_column(df_clean, col)
    
    return df_clean

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Basic validation of DataFrame structure.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        True if DataFrame is valid
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if df.empty:
        return False
    
    if df.columns.duplicated().any():
        return False
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 3, 4],
        'value': [10, 20, None, 30, 40],
        'score': [85, 92, 78, 78, 88]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataset(df, normalize_cols=['score'])
    print(cleaned_df)