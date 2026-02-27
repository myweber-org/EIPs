import numpy as np
import pandas as pd

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to analyze
        threshold: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outlier rows
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def remove_duplicates(data, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        data: pandas DataFrame
        subset: columns to consider for duplicates
        keep: which duplicates to keep ('first', 'last', False)
    
    Returns:
        DataFrame with duplicates removed
    """
    return data.drop_duplicates(subset=subset, keep=keep)

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: specific columns to process
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = data.columns
    
    data_copy = data.copy()
    
    for col in columns:
        if col not in data_copy.columns:
            continue
            
        if strategy == 'drop':
            data_copy = data_copy.dropna(subset=[col])
        elif strategy == 'mean':
            data_copy[col] = data_copy[col].fillna(data_copy[col].mean())
        elif strategy == 'median':
            data_copy[col] = data_copy[col].fillna(data_copy[col].median())
        elif strategy == 'mode':
            data_copy[col] = data_copy[col].fillna(data_copy[col].mode()[0])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return data_copy

def validate_data_types(data, schema):
    """
    Validate DataFrame columns against expected data types.
    
    Args:
        data: pandas DataFrame
        schema: dictionary mapping column names to expected types
    
    Returns:
        Dictionary with validation results
    """
    results = {}
    
    for column, expected_type in schema.items():
        if column not in data.columns:
            results[column] = {'valid': False, 'message': 'Column not found'}
            continue
        
        actual_type = str(data[column].dtype)
        
        if expected_type == 'numeric':
            valid = np.issubdtype(data[column].dtype, np.number)
        elif expected_type == 'categorical':
            valid = data[column].dtype == 'object'
        elif expected_type == 'datetime':
            valid = np.issubdtype(data[column].dtype, np.datetime64)
        else:
            valid = actual_type == expected_type
        
        results[column] = {
            'valid': valid,
            'expected': expected_type,
            'actual': actual_type
        }
    
    return results