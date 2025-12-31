import pandas as pd
import numpy as np
from typing import Optional, List

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

def fill_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fill missing values in DataFrame columns.
    
    Args:
        df: Input DataFrame
        strategy: 'mean', 'median', 'mode', or 'constant'
        columns: Specific columns to fill, fills all if None
    
    Returns:
        DataFrame with missing values filled
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.columns
    
    for col in columns:
        if df_copy[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif strategy == 'median':
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif strategy == 'mode':
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
            elif strategy == 'constant':
                df_copy[col].fillna(0, inplace=True)
        else:
            df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
    
    return df_copy

def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalize a numeric column to range [0, 1].
    
    Args:
        df: Input DataFrame
        column: Column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    df_copy = df.copy()
    if df_copy[column].dtype in ['int64', 'float64']:
        col_min = df_copy[column].min()
        col_max = df_copy[column].max()
        if col_max != col_min:
            df_copy[column] = (df_copy[column] - col_min) / (col_max - col_min)
    return df_copy

def remove_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers using IQR method.
    
    Args:
        df: Input DataFrame
        column: Column to check for outliers
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if df[column].dtype not in ['int64', 'float64']:
        return df
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataframe(df: pd.DataFrame, 
                   remove_dups: bool = True,
                   fill_na: bool = True,
                   fill_strategy: str = 'mean',
                   normalize_cols: Optional[List[str]] = None,
                   remove_outlier_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        remove_dups: Whether to remove duplicates
        fill_na: Whether to fill missing values
        fill_strategy: Strategy for filling missing values
        normalize_cols: Columns to normalize
        remove_outlier_cols: Columns to remove outliers from
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if fill_na:
        cleaned_df = fill_missing_values(cleaned_df, strategy=fill_strategy)
    
    if normalize_cols:
        for col in normalize_cols:
            if col in cleaned_df.columns:
                cleaned_df = normalize_column(cleaned_df, col)
    
    if remove_outlier_cols:
        for col in remove_outlier_cols:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    return cleaned_df

def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Validate DataFrame and return statistics.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dictionary with validation statistics
    """
    stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'column_stats': {}
    }
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            stats['column_stats'][col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'unique_values': df[col].nunique()
            }
        else:
            stats['column_stats'][col] = {
                'unique_values': df[col].nunique(),
                'most_common': df[col].mode().iloc[0] if not df[col].mode().empty else None
            }
    
    return stats

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, 4, 5, None, 7, 8, 9, 100],
        'B': [10, 20, 20, 40, 50, 60, 70, 80, 90, 1000],
        'C': ['a', 'b', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation Stats:")
    print(validate_dataframe(df))
    
    cleaned = clean_dataframe(df, 
                             normalize_cols=['A', 'B'],
                             remove_outlier_cols=['A', 'B'])
    
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nCleaned Validation Stats:")
    print(validate_dataframe(cleaned))