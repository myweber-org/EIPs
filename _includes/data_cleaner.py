
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

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling.
    
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

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: list of columns to process (None for all numeric columns)
    
    Returns:
        DataFrame with missing values handled
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    data_copy = data.copy()
    
    for col in columns:
        if col not in data_copy.columns:
            continue
            
        if strategy == 'drop':
            data_copy = data_copy.dropna(subset=[col])
        elif strategy == 'mean':
            data_copy[col].fillna(data_copy[col].mean(), inplace=True)
        elif strategy == 'median':
            data_copy[col].fillna(data_copy[col].median(), inplace=True)
        elif strategy == 'mode':
            mode_val = data_copy[col].mode()
            if not mode_val.empty:
                data_copy[col].fillna(mode_val[0], inplace=True)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return data_copy

def create_data_summary(data):
    """
    Create comprehensive summary statistics for DataFrame.
    
    Args:
        data: pandas DataFrame
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'shape': data.shape,
        'columns': list(data.columns),
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'numeric_summary': {},
        'categorical_summary': {}
    }
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_summary'][col] = {
            'mean': data[col].mean(),
            'std': data[col].std(),
            'min': data[col].min(),
            '25%': data[col].quantile(0.25),
            '50%': data[col].median(),
            '75%': data[col].quantile(0.75),
            'max': data[col].max(),
            'skewness': data[col].skew(),
            'kurtosis': data[col].kurtosis()
        }
    
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        value_counts = data[col].value_counts()
        summary['categorical_summary'][col] = {
            'unique_count': data[col].nunique(),
            'top_value': value_counts.index[0] if not value_counts.empty else None,
            'top_count': value_counts.iloc[0] if not value_counts.empty else 0,
            'value_distribution': value_counts.head(10).to_dict()
        }
    
    return summary

def detect_duplicates(data, subset=None, keep='first'):
    """
    Detect and handle duplicate rows.
    
    Args:
        data: pandas DataFrame
        subset: columns to consider for duplicates
        keep: which duplicates to mark ('first', 'last', False)
    
    Returns:
        Tuple of (DataFrame without duplicates, duplicate indices)
    """
    duplicates = data.duplicated(subset=subset, keep=keep)
    duplicate_indices = data.index[duplicates].tolist()
    
    if keep == False:
        # Remove all duplicates
        cleaned_data = data.drop_duplicates(subset=subset, keep=False)
    else:
        # Keep first/last occurrence
        cleaned_data = data.drop_duplicates(subset=subset, keep=keep)
    
    return cleaned_data, duplicate_indices