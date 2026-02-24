
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data
    
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        DataFrame with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data
    
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_missing_values(data, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Args:
        data: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
    
    Returns:
        DataFrame with missing values handled
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        return data.dropna(subset=numeric_cols)
    
    for col in numeric_cols:
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            data[col] = data[col].fillna(fill_value)
    
    return data

def process_dataframe(df, numeric_columns=None, outlier_removal=True, 
                     normalization='minmax', missing_strategy='mean'):
    """
    Complete data cleaning pipeline.
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric columns to process (None for all numeric)
        outlier_removal: Whether to remove outliers
        normalization: 'minmax', 'zscore', or None
        missing_strategy: Strategy for handling missing values
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    result_df = df.copy()
    
    # Handle missing values
    result_df = clean_missing_values(result_df, strategy=missing_strategy)
    
    # Remove outliers for each numeric column
    if outlier_removal:
        for col in numeric_columns:
            if col in result_df.columns:
                result_df = remove_outliers_iqr(result_df, col)
    
    # Apply normalization
    if normalization == 'minmax':
        for col in numeric_columns:
            if col in result_df.columns:
                result_df = normalize_minmax(result_df, col)
    elif normalization == 'zscore':
        for col in numeric_columns:
            if col in result_df.columns:
                result_df = standardize_zscore(result_df, col)
    
    return result_df