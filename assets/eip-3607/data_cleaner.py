
import re
import pandas as pd
from typing import List, Optional

def remove_special_chars(text: str) -> str:
    """Remove non-alphanumeric characters from a string."""
    if not isinstance(text, str):
        return text
    return re.sub(r'[^A-Za-z0-9\s]+', '', text)

def normalize_whitespace(text: str) -> str:
    """Replace multiple whitespace characters with a single space."""
    if not isinstance(text, str):
        return text
    return re.sub(r'\s+', ' ', text).strip()

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names by removing special characters and normalizing."""
    df.columns = [remove_special_chars(str(col)) for col in df.columns]
    df.columns = [normalize_whitespace(col) for col in df.columns]
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    return df

def drop_missing_rows(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Drop rows with missing values in specified columns or all columns."""
    if columns:
        return df.dropna(subset=columns)
    return df.dropna()

def convert_to_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Convert specified columns to numeric, coercing errors to NaN."""
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    return df_copy

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a series of cleaning functions to a DataFrame."""
    df = clean_column_names(df)
    df = drop_missing_rows(df)
    return df
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column in a DataFrame using the IQR method.
    
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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by removing outliers from all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names. If None, uses all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.concatenate([
            np.random.normal(100, 10, 90),
            np.random.normal(300, 50, 10)
        ]),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Original dataset shape:", df.shape)
    print("\nOriginal statistics:")
    print(calculate_basic_stats(df, 'value'))
    
    cleaned_df = clean_dataset(df, ['value'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned statistics:")
    print(calculate_basic_stats(cleaned_df, 'value'))