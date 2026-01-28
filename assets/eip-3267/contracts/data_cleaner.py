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
        columns: list of columns to apply strategy to
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = df.columns
    
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns:
            if strategy == 'mean':
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif strategy == 'median':
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif strategy == 'mode':
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_copy.dropna(subset=[col], inplace=True)
    
    return df_copy

def normalize_column(df, column, method='minmax'):
    """
    Normalize values in a column.
    
    Args:
        df: pandas DataFrame
        column: column name to normalize
        method: 'minmax' or 'zscore'
    
    Returns:
        DataFrame with normalized column
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
    
    return df_copy

def detect_outliers(df, column, method='iqr', threshold=1.5):
    """
    Detect outliers in a column.
    
    Args:
        df: pandas DataFrame
        column: column name to check for outliers
        method: 'iqr' or 'zscore'
        threshold: threshold value for outlier detection
    
    Returns:
        Boolean Series indicating outliers
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return z_scores > threshold
    
    return pd.Series([False] * len(df), index=df.index)

def clean_dataframe(df, config):
    """
    Apply multiple cleaning operations based on configuration.
    
    Args:
        df: pandas DataFrame
        config: dictionary with cleaning configuration
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if config.get('remove_duplicates', False):
        subset = config.get('duplicate_subset', None)
        df_clean = remove_duplicates(df_clean, subset)
    
    if 'missing_values' in config:
        strategy = config['missing_values'].get('strategy', 'mean')
        columns = config['missing_values'].get('columns', None)
        df_clean = handle_missing_values(df_clean, strategy, columns)
    
    if 'normalize' in config:
        for col_config in config['normalize']:
            column = col_config['column']
            method = col_config.get('method', 'minmax')
            df_clean = normalize_column(df_clean, column, method)
    
    return df_clean