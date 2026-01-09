import pandas as pd
import numpy as np
from scipy import stats

def normalize_data(df, columns=None, method='zscore'):
    """
    Normalize specified columns in DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to normalize (default: all numeric columns)
        method: normalization method ('zscore', 'minmax', or 'robust')
    
    Returns:
        Normalized DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_normalized = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'zscore':
            df_normalized[col] = stats.zscore(df[col])
        elif method == 'minmax':
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max != col_min:
                df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
        elif method == 'robust':
            col_median = df[col].median()
            col_iqr = stats.iqr(df[col])
            if col_iqr > 0:
                df_normalized[col] = (df[col] - col_median) / col_iqr
    
    return df_normalized

def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Remove outliers from specified columns.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to check for outliers
        method: outlier detection method ('iqr' or 'zscore')
        threshold: threshold for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    mask = pd.Series([True] * len(df))
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col].fillna(df[col].mean())))
            col_mask = z_scores < threshold
        
        mask = mask & col_mask
    
    return df_clean[mask]

def handle_missing_values(df, columns=None, strategy='mean'):
    """
    Handle missing values in specified columns.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to handle missing values
        strategy: imputation strategy ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = df.columns.tolist()
    
    df_processed = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'drop':
            df_processed = df_processed.dropna(subset=[col])
        elif strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
        elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        elif strategy == 'mode':
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0] if not df_processed[col].mode().empty else None)
    
    return df_processed

def clean_dataset(df, config=None):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        config: dictionary with cleaning configuration
    
    Returns:
        Cleaned DataFrame
    """
    if config is None:
        config = {
            'missing_values': {'strategy': 'mean'},
            'normalization': {'method': 'zscore'},
            'outliers': {'method': 'iqr', 'threshold': 1.5}
        }
    
    df_clean = df.copy()
    
    # Handle missing values
    if 'missing_values' in config:
        df_clean = handle_missing_values(
            df_clean, 
            strategy=config['missing_values'].get('strategy', 'mean')
        )
    
    # Remove outliers
    if 'outliers' in config:
        df_clean = remove_outliers(
            df_clean,
            method=config['outliers'].get('method', 'iqr'),
            threshold=config['outliers'].get('threshold', 1.5)
        )
    
    # Normalize data
    if 'normalization' in config:
        df_clean = normalize_data(
            df_clean,
            method=config['normalization'].get('method', 'zscore')
        )
    
    return df_clean