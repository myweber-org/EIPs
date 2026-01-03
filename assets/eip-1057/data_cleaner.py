import pandas as pd
import numpy as np
from typing import Optional, List

def remove_duplicate_rows(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fill missing values using specified strategy.
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if df[col].dtype in [np.float64, np.int64]:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
            else:
                fill_value = 0
            
            df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalize a column to range [0, 1].
    """
    df_normalized = df.copy()
    col_min = df[column].min()
    col_max = df[column].max()
    
    if col_max != col_min:
        df_normalized[column] = (df[column] - col_min) / (col_max - col_min)
    
    return df_normalized

def clean_csv_file(input_path: str, output_path: str, remove_dups: bool = True, fill_na: bool = True) -> None:
    """
    Main function to clean CSV file.
    """
    df = pd.read_csv(input_path)
    
    if remove_dups:
        df = remove_duplicate_rows(df)
    
    if fill_na:
        df = fill_missing_values(df, strategy='mean')
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")