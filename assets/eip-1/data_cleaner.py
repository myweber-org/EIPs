
import pandas as pd
import numpy as np

def clean_missing_data(df, method='mean', columns=None):
    """
    Clean missing values in a DataFrame using specified method.
    
    Args:
        df: pandas DataFrame containing data with potential missing values
        method: Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
        columns: List of columns to apply cleaning to, or None for all columns
    
    Returns:
        Cleaned DataFrame with missing values handled
    """
    if df.empty:
        return df
    
    if columns is None:
        columns = df.columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if df_clean[col].isnull().sum() == 0:
            continue
        
        if method == 'mean':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        
        elif method == 'median':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        elif method == 'mode':
            if not df_clean[col].mode().empty:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        elif method == 'drop':
            df_clean = df_clean.dropna(subset=[col])
    
    return df_clean

def detect_outliers(df, column, method='iqr', threshold=1.5):
    """
    Detect outliers in a specific column using specified method.
    
    Args:
        df: pandas DataFrame
        column: Column name to check for outliers
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
    
    Returns:
        Boolean mask indicating outliers (True for outliers)
    """
    if column not in df.columns:
        return pd.Series([False] * len(df))
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == 'zscore':
        mean = df[column].mean()
        std = df[column].std()
        if std == 0:
            return pd.Series([False] * len(df))
        z_scores = np.abs((df[column] - mean) / std)
        return z_scores > threshold
    
    return pd.Series([False] * len(df))

def normalize_column(df, column, method='minmax'):
    """
    Normalize values in a specific column.
    
    Args:
        df: pandas DataFrame
        column: Column name to normalize
        method: Normalization method ('minmax' or 'standard')
    
    Returns:
        Series with normalized values
    """
    if column not in df.columns:
        return pd.Series()
    
    if method == 'minmax':
        col_min = df[column].min()
        col_max = df[column].max()
        if col_max == col_min:
            return pd.Series([0] * len(df))
        return (df[column] - col_min) / (col_max - col_min)
    
    elif method == 'standard':
        col_mean = df[column].mean()
        col_std = df[column].std()
        if col_std == 0:
            return pd.Series([0] * len(df))
        return (df[column] - col_mean) / col_std
    
    return df[column].copy()

def validate_dataframe(df, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of columns that must be present
        numeric_columns: List of columns that must be numeric
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None:
        return False, "DataFrame is None"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    if numeric_columns:
        non_numeric_cols = []
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_cols.append(col)
        if non_numeric_cols:
            return False, f"Non-numeric columns: {non_numeric_cols}"
    
    return True, "DataFrame is valid"