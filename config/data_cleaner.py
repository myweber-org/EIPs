
import numpy as np
import pandas as pd
from scipy import stats

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
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_indices = np.where(z_scores < threshold)[0]
    filtered_data = data.iloc[filtered_indices].copy()
    return filtered_data

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
    
    normalized_data = data.copy()
    min_val = normalized_data[column].min()
    max_val = normalized_data[column].max()
    
    if max_val != min_val:
        normalized_data[column] = (normalized_data[column] - min_val) / (max_val - min_val)
    else:
        normalized_data[column] = 0.5
    
    return normalized_data

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    standardized_data = data.copy()
    mean_val = standardized_data[column].mean()
    std_val = standardized_data[column].std()
    
    if std_val > 0:
        standardized_data[column] = (standardized_data[column] - mean_val) / std_val
    else:
        standardized_data[column] = 0
    
    return standardized_data

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names to process
        outlier_method: 'iqr' or 'zscore' (default 'iqr')
        normalize_method: 'minmax' or 'zscore' (default 'minmax')
    
    Returns:
        Cleaned and normalized DataFrame
    """
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown outlier method: {outlier_method}")
        
        if normalize_method == 'minmax':
            cleaned_data = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data = normalize_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown normalize method: {normalize_method}")
    
    return cleaned_data.reset_index(drop=True)

def validate_data(data, required_columns, numeric_check=True):
    """
    Validate dataset structure and content.
    
    Args:
        data: pandas DataFrame to validate
        required_columns: list of required column names
        numeric_check: whether to check for numeric columns
    
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'missing_columns': [],
        'non_numeric_columns': [],
        'null_counts': {},
        'row_count': len(data),
        'column_count': len(data.columns)
    }
    
    for column in required_columns:
        if column not in data.columns:
            validation_result['missing_columns'].append(column)
            validation_result['is_valid'] = False
    
    if numeric_check:
        for column in data.select_dtypes(include=[np.number]).columns:
            if not pd.api.types.is_numeric_dtype(data[column]):
                validation_result['non_numeric_columns'].append(column)
                validation_result['is_valid'] = False
    
    for column in data.columns:
        null_count = data[column].isnull().sum()
        if null_count > 0:
            validation_result['null_counts'][column] = null_count
    
    return validation_result