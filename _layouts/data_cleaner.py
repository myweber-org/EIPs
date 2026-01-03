
import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicates(df: pd.DataFrame, 
                      subset: Optional[List[str]] = None,
                      keep: str = 'first') -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df: Input DataFrame
    subset: Columns to consider for identifying duplicates
    keep: Which duplicates to keep - 'first', 'last', or False
    
    Returns:
    Cleaned DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    if subset is not None:
        missing_cols = [col for col in subset if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Perform basic validation on DataFrame.
    
    Parameters:
    df: DataFrame to validate
    
    Returns:
    Boolean indicating if DataFrame is valid
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if df.isnull().all().any():
        print("Warning: Some columns contain only null values")
    
    return True

def clean_numeric_columns(df: pd.DataFrame, 
                         columns: List[str]) -> pd.DataFrame:
    """
    Clean numeric columns by converting to appropriate types.
    
    Parameters:
    df: Input DataFrame
    columns: List of column names to clean
    
    Returns:
    DataFrame with cleaned numeric columns
    """
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    return cleaned_df

def main():
    """
    Example usage of data cleaning functions.
    """
    sample_data = {
        'id': [1, 2, 3, 1, 2, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob', 'David'],
        'score': [85, 92, 78, 85, 92, 88],
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', 
                '2023-01-01', '2023-01-02', '2023-01-04']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = remove_duplicates(df, subset=['id', 'name'], keep='first')
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    if validate_dataframe(cleaned_df):
        print("DataFrame validation passed")

if __name__ == "__main__":
    main()