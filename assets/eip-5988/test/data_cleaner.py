
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
    keep: Which duplicates to keep ('first', 'last', False)
    
    Returns:
    DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    if subset is None:
        subset = df.columns.tolist()
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df.reset_index(drop=True)

def clean_numeric_columns(df: pd.DataFrame, 
                         columns: List[str]) -> pd.DataFrame:
    """
    Clean numeric columns by converting to appropriate types
    and handling invalid values.
    """
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
    
    return df_clean

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Perform basic validation on DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    return True

def get_cleaning_summary(df_before: pd.DataFrame, 
                        df_after: pd.DataFrame) -> dict:
    """
    Generate a summary of data cleaning operations.
    """
    summary = {
        'rows_removed': len(df_before) - len(df_after),
        'columns_removed': len(df_before.columns) - len(df_after.columns),
        'final_shape': df_after.shape,
        'null_values': df_after.isnull().sum().sum()
    }
    
    return summary