
import pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy to handle missing values ('mean', 'median', 'mode', 'drop')
        columns (list): List of columns to apply cleaning, if None applies to all numeric columns
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if strategy == 'mean':
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        elif strategy == 'median':
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif strategy == 'mode':
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        elif strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
    
    return df_clean

def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Remove outliers from DataFrame using specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to check for outliers
        method (str): Method to detect outliers ('iqr' or 'zscore')
        threshold (float): Threshold for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    mask = pd.Series([True] * len(df_clean))
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            col_mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            col_mask = z_scores <= threshold
        
        mask = mask & col_mask
    
    return df_clean[mask]

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize data in specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to normalize
        method (str): Normalization method ('minmax' or 'standard')
    
    Returns:
        pd.DataFrame: DataFrame with normalized columns
    """
    df_norm = df.copy()
    
    if columns is None:
        numeric_cols = df_norm.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col not in df_norm.columns:
            continue
            
        if method == 'minmax':
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val != min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        elif method == 'standard':
            mean_val = df_norm[col].mean()
            std_val = df_norm[col].std()
            if std_val != 0:
                df_norm[col] = (df_norm[col] - mean_val) / std_val
    
    return df_norm

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr', normalize=False):
    """
    Complete data cleaning pipeline.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_strategy (str): Strategy for handling missing values
        outlier_method (str): Method for outlier detection
        normalize (bool): Whether to normalize the data
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = clean_missing_values(df, strategy=missing_strategy)
    df_clean = remove_outliers(df_clean, method=outlier_method)
    
    if normalize:
        df_clean = normalize_data(df_clean)
    
    return df_clean