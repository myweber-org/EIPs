
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        column (str): Column name to clean
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return filtered_data

def calculate_summary_statistics(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    if data.empty:
        return {}
    
    stats = {
        'mean': np.mean(data[column]),
        'median': np.median(data[column]),
        'std': np.std(data[column]),
        'min': np.min(data[column]),
        'max': np.max(data[column]),
        'count': len(data[column])
    }
    
    return statsimport pandas as pd
import numpy as np
from typing import Union, List, Optional

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df: pd.DataFrame, column: str, fill_value: Union[str, int, float] = None) -> pd.DataFrame:
    """
    Fill missing values in a specified column.
    """
    df_copy = df.copy()
    if fill_value is not None:
        df_copy[column] = df_copy[column].fillna(fill_value)
    else:
        if pd.api.types.is_numeric_dtype(df_copy[column]):
            df_copy[column] = df_copy[column].fillna(df_copy[column].median())
        else:
            df_copy[column] = df_copy[column].fillna(df_copy[column].mode()[0])
    return df_copy

def validate_email(email_series: pd.Series) -> pd.Series:
    """
    Validate email addresses using a simple regex pattern.
    Returns a boolean Series.
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return email_series.str.match(pattern, na=False)

def convert_to_datetime(df: pd.DataFrame, column: str, format: str = None) -> pd.DataFrame:
    """
    Convert a column to datetime format.
    """
    df_copy = df.copy()
    if format:
        df_copy[column] = pd.to_datetime(df_copy[column], format=format, errors='coerce')
    else:
        df_copy[column] = pd.to_datetime(df_copy[column], errors='coerce')
    return df_copy

def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalize a numeric column to range [0, 1].
    """
    df_copy = df.copy()
    if pd.api.types.is_numeric_dtype(df_copy[column]):
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val > min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    return df_copy

def clean_dataframe(df: pd.DataFrame, 
                    drop_duplicates: bool = True,
                    fill_missing: Optional[List[str]] = None,
                    datetime_columns: Optional[dict] = None,
                    normalize_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Apply multiple cleaning operations to a DataFrame.
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        df_clean = remove_duplicates(df_clean)
    
    if fill_missing:
        for col in fill_missing:
            df_clean = fill_missing_values(df_clean, col)
    
    if datetime_columns:
        for col, fmt in datetime_columns.items():
            df_clean = convert_to_datetime(df_clean, col, fmt)
    
    if normalize_columns:
        for col in normalize_columns:
            df_clean = normalize_column(df_clean, col)
    
    return df_clean