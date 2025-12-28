
def remove_duplicates(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def normalize_minmax(data, columns=None):
    """
    Normalize data using Min-Max scaling
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    normalized_data = data.copy()
    
    for col in columns:
        if col in data.columns and np.issubdtype(data[col].dtype, np.number):
            min_val = data[col].min()
            max_val = data[col].max()
            
            if max_val != min_val:
                normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
            else:
                normalized_data[col] = 0
    
    return normalized_data

def normalize_zscore(data, columns=None):
    """
    Normalize data using Z-score standardization
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    standardized_data = data.copy()
    
    for col in columns:
        if col in data.columns and np.issubdtype(data[col].dtype, np.number):
            mean_val = data[col].mean()
            std_val = data[col].std()
            
            if std_val != 0:
                standardized_data[col] = (data[col] - mean_val) / std_val
            else:
                standardized_data[col] = 0
    
    return standardized_data

def clean_dataset(data, outlier_method='iqr', outlier_columns=None, 
                  normalize_method='minmax', normalize_columns=None):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_data = data.copy()
    
    if outlier_columns is None:
        outlier_columns = data.select_dtypes(include=[np.number]).columns
    
    outliers_summary = {}
    
    for col in outlier_columns:
        if col in data.columns and np.issubdtype(data[col].dtype, np.number):
            if outlier_method == 'iqr':
                cleaned_data, removed = remove_outliers_iqr(cleaned_data, col)
            elif outlier_method == 'zscore':
                cleaned_data, removed = remove_outliers_zscore(cleaned_data, col)
            else:
                raise ValueError(f"Unknown outlier method: {outlier_method}")
            
            outliers_summary[col] = removed
    
    if normalize_method == 'minmax':
        cleaned_data = normalize_minmax(cleaned_data, normalize_columns)
    elif normalize_method == 'zscore':
        cleaned_data = normalize_zscore(cleaned_data, normalize_columns)
    elif normalize_method is not None:
        raise ValueError(f"Unknown normalize method: {normalize_method}")
    
    return cleaned_data, outliers_summary

def validate_data(data, check_missing=True, check_duplicates=True):
    """
    Validate data quality
    """
    validation_report = {}
    
    if check_missing:
        missing_values = data.isnull().sum()
        missing_percentage = (missing_values / len(data)) * 100
        validation_report['missing_values'] = missing_values[missing_values > 0].to_dict()
        validation_report['missing_percentage'] = missing_percentage[missing_percentage > 0].to_dict()
    
    if check_duplicates:
        duplicate_count = data.duplicated().sum()
        validation_report['duplicate_rows'] = duplicate_count
        validation_report['duplicate_percentage'] = (duplicate_count / len(data)) * 100
    
    validation_report['total_rows'] = len(data)
    validation_report['total_columns'] = len(data.columns)
    validation_report['data_types'] = data.dtypes.to_dict()
    
    return validation_report