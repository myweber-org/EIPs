import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: list of columns to apply the strategy to
    
    Returns:
        DataFrame with missing values handled
    """
    if columns is None:
        columns = df.columns
    
    df_copy = df.copy()
    
    for col in columns:
        if col in df.columns:
            if strategy == 'mean':
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif strategy == 'median':
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif strategy == 'mode':
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_copy = df_copy.dropna(subset=[col])
    
    return df_copy

def normalize_columns(df, columns=None):
    """
    Normalize specified columns to range [0, 1].
    
    Args:
        df: pandas DataFrame
        columns: list of columns to normalize
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_copy = df.copy()
    
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            min_val = df_copy[col].min()
            max_val = df_copy[col].max()
            
            if max_val > min_val:
                df_copy[col] = (df_copy[col] - min_val) / (max_val - min_val)
    
    return df_copy

def detect_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Detect outliers using IQR method.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to check for outliers
        threshold: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outlier flags
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    outlier_flags = pd.DataFrame(index=df.index)
    
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_flags[f'{col}_outlier'] = ~df[col].between(lower_bound, upper_bound)
    
    return outlier_flags

def clean_dataset(df, remove_dups=True, handle_nan=True, normalize=True):
    """
    Comprehensive dataset cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        remove_dups: whether to remove duplicates
        handle_nan: whether to handle missing values
        normalize: whether to normalize numeric columns
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if handle_nan:
        cleaned_df = handle_missing_values(cleaned_df, strategy='mean')
    
    if normalize:
        cleaned_df = normalize_columns(cleaned_df)
    
    return cleaned_df