import pandas as pd
import numpy as np
from typing import Optional, List, Union

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

def handle_missing_values(df: pd.DataFrame, 
                         strategy: str = 'drop', 
                         fill_value: Union[int, float, str] = None) -> pd.DataFrame:
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
    return df

def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalize a numeric column to range [0, 1].
    
    Args:
        df: Input DataFrame
        column: Column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
        col_min = df[column].min()
        col_max = df[column].max()
        
        if col_max != col_min:
            df[column] = (df[column] - col_min) / (col_max - col_min)
    
    return df

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names by removing whitespace and converting to lowercase.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with cleaned column names
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def validate_dataframe(df: pd.DataFrame, 
                      required_columns: List[str] = None) -> bool:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
    
    Returns:
        Boolean indicating if DataFrame is valid
    """
    if df.empty:
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False
    
    return True

def process_csv_file(filepath: str, 
                    cleaning_steps: List[str] = None) -> pd.DataFrame:
    """
    Main function to process CSV file with cleaning steps.
    
    Args:
        filepath: Path to CSV file
        cleaning_steps: List of cleaning steps to apply
    
    Returns:
        Cleaned DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        
        if cleaning_steps is None:
            cleaning_steps = ['clean_names', 'remove_duplicates', 'handle_missing']
        
        for step in cleaning_steps:
            if step == 'clean_names':
                df = clean_column_names(df)
            elif step == 'remove_duplicates':
                df = remove_duplicates(df)
            elif step == 'handle_missing':
                df = handle_missing_values(df, strategy='fill')
        
        return df
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return pd.DataFrame()