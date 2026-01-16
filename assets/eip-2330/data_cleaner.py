import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def zscore_normalize(data, column):
    """
    Normalize data using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    data_copy = data.copy()
    mean = data_copy[column].mean()
    std = data_copy[column].std()
    
    if std == 0:
        data_copy[f'{column}_normalized'] = 0
    else:
        data_copy[f'{column}_normalized'] = (data_copy[column] - mean) / std
    
    return data_copy

def minmax_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize data using min-max normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
        feature_range: desired range of transformed data (default 0-1)
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    data_copy = data.copy()
    min_val = data_copy[column].min()
    max_val = data_copy[column].max()
    
    if min_val == max_val:
        data_copy[f'{column}_normalized'] = feature_range[0]
    else:
        data_copy[f'{column}_normalized'] = (
            (data_copy[column] - min_val) / (max_val - min_val) * 
            (feature_range[1] - feature_range[0]) + feature_range[0]
        )
    
    return data_copy

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: list of columns to process (None for all numeric columns)
    
    Returns:
        DataFrame with handled missing values
    """
    data_copy = data.copy()
    
    if columns is None:
        columns = data_copy.select_dtypes(include=[np.number]).columns
    
    for column in columns:
        if column not in data_copy.columns:
            continue
            
        if strategy == 'drop':
            data_copy = data_copy.dropna(subset=[column])
        elif strategy == 'mean':
            data_copy[column] = data_copy[column].fillna(data_copy[column].mean())
        elif strategy == 'median':
            data_copy[column] = data_copy[column].fillna(data_copy[column].median())
        elif strategy == 'mode':
            mode_val = data_copy[column].mode()
            if not mode_val.empty:
                data_copy[column] = data_copy[column].fillna(mode_val[0])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return data_copy

def validate_data(data, check_negative=False, check_zero=False):
    """
    Validate data for common issues.
    
    Args:
        data: pandas DataFrame
        check_negative: flag to check for negative values
        check_zero: flag to check for zero values
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'has_nulls': data.isnull().any().any(),
        'null_columns': data.columns[data.isnull().any()].tolist(),
        'null_count': data.isnull().sum().sum(),
        'duplicate_rows': data.duplicated().sum()
    }
    
    if check_negative:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        negative_cols = []
        for col in numeric_cols:
            if (data[col] < 0).any():
                negative_cols.append(col)
        validation_results['negative_columns'] = negative_cols
    
    if check_zero:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        zero_cols = []
        for col in numeric_cols:
            if (data[col] == 0).any():
                zero_cols.append(col)
        validation_results['zero_columns'] = zero_cols
    
    return validation_results