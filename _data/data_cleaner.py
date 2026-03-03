
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def normalize_column(df, column, method='minmax'):
    """
    Normalize a DataFrame column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val != 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

def handle_missing_values(df, column, strategy='mean'):
    """
    Handle missing values in a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if strategy == 'mean':
        fill_value = df_copy[column].mean()
    elif strategy == 'median':
        fill_value = df_copy[column].median()
    elif strategy == 'mode':
        fill_value = df_copy[column].mode()[0]
    elif strategy == 'drop':
        df_copy = df_copy.dropna(subset=[column])
        return df_copy
    else:
        raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
    
    df_copy[column] = df_copy[column].fillna(fill_value)
    return df_copy
import re
import pandas as pd
from typing import Union, List, Optional

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names: lowercase, replace spaces with underscores.
    """
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    return df

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def validate_email(email: str) -> bool:
    """
    Validate email format using regex.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def fill_missing_values(df: pd.DataFrame, column: str, value: Union[str, int, float]) -> pd.DataFrame:
    """
    Fill missing values in a specified column with a given value.
    """
    df[column] = df[column].fillna(value)
    return df

def convert_to_datetime(df: pd.DataFrame, column: str, format: str = '%Y-%m-%d') -> pd.DataFrame:
    """
    Convert a column to datetime format.
    """
    df[column] = pd.to_datetime(df[column], format=format, errors='coerce')
    return df

def filter_by_threshold(df: pd.DataFrame, column: str, threshold: float, keep: str = 'above') -> pd.DataFrame:
    """
    Filter rows based on a numeric threshold.
    """
    if keep == 'above':
        return df[df[column] > threshold]
    elif keep == 'below':
        return df[df[column] < threshold]
    else:
        raise ValueError("keep parameter must be 'above' or 'below'")

def main():
    """
    Example usage of data cleaning functions.
    """
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', None],
        'Email': ['alice@example.com', 'invalid-email', 'charlie@test.org', 'david@domain.net'],
        'Score': [85.5, 92.0, 78.5, 88.0],
        'Join Date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05']
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    df = clean_column_names(df)
    df = fill_missing_values(df, 'name', 'Unknown')
    df = convert_to_datetime(df, 'join_date')
    df = filter_by_threshold(df, 'score', 80.0, keep='above')
    
    print("\nCleaned DataFrame:")
    print(df)
    
    email_check = df['email'].apply(validate_email)
    print("\nValid emails:", df[email_check]['email'].tolist())

if __name__ == '__main__':
    main()