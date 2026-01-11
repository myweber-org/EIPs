
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
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (bool): Whether to fill missing values.
    fill_value: Value to use for filling missing data.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if df.empty:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to process.
    multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df