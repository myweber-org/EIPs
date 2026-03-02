import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using the Interquartile Range method.
    Returns filtered DataFrame and outlier indices.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    mask = (data[column] >= lower_bound) & (data[column] <= upper_bound)
    outliers = data[~mask].index.tolist()
    cleaned_data = data[mask].copy()
    
    return cleaned_data, outliers

def normalize_minmax(data, columns=None):
    """
    Apply Min-Max normalization to specified columns.
    If columns is None, normalize all numeric columns.
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    normalized_data = data.copy()
    for col in columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            col_min = normalized_data[col].min()
            col_max = normalized_data[col].max()
            if col_max > col_min:
                normalized_data[col] = (normalized_data[col] - col_min) / (col_max - col_min)
    
    return normalized_data

def z_score_normalize(data, columns=None):
    """
    Apply Z-score normalization to specified columns.
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    normalized_data = data.copy()
    for col in columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            mean_val = normalized_data[col].mean()
            std_val = normalized_data[col].std()
            if std_val > 0:
                normalized_data[col] = (normalized_data[col] - mean_val) / std_val
    
    return normalized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    Supported strategies: 'mean', 'median', 'mode', 'drop'
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    if strategy == 'drop':
        cleaned_data = cleaned_data.dropna(subset=columns)
    else:
        for col in columns:
            if col in cleaned_data.columns and cleaned_data[col].isnull().any():
                if strategy == 'mean':
                    fill_value = cleaned_data[col].mean()
                elif strategy == 'median':
                    fill_value = cleaned_data[col].median()
                elif strategy == 'mode':
                    fill_value = cleaned_data[col].mode()[0]
                else:
                    raise ValueError(f"Unsupported strategy: {strategy}")
                
                cleaned_data[col] = cleaned_data[col].fillna(fill_value)
    
    return cleaned_data

def validate_dataframe(data, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and data types.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if numeric_columns:
        non_numeric = [col for col in numeric_columns if not pd.api.types.is_numeric_dtype(data[col])]
        if non_numeric:
            raise ValueError(f"Non-numeric columns specified as numeric: {non_numeric}")
    
    return True