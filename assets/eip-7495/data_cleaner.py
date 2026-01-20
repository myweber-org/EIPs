
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
    
    return df_cleanimport pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fill_missing (str or dict): Method to fill missing values.
            Options: 'mean', 'median', 'mode', or a dictionary of column:value pairs.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"