
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
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

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
    mask = z_scores < threshold
    
    valid_indices = data[column].dropna().index[mask]
    return data.loc[valid_indices]

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
    
    col_data = data[column].copy()
    min_val = col_data.min()
    max_val = col_data.max()
    
    if max_val == min_val:
        return col_data * 0
    
    return (col_data - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    col_data = data[column].copy()
    mean_val = col_data.mean()
    std_val = col_data.std()
    
    if std_val == 0:
        return col_data * 0
    
    return (col_data - mean_val) / std_val

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names (if None, auto-detect)
        outlier_method: 'iqr' or 'zscore' (default 'iqr')
        normalize_method: 'minmax' or 'zscore' (default 'minmax')
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
        
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        
        if normalize_method == 'minmax':
            cleaned_data[f'{column}_normalized'] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[f'{column}_normalized'] = normalize_zscore(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, required_columns=None, allow_nan_ratio=0.1):
    """
    Validate dataset structure and quality.
    
    Args:
        data: pandas DataFrame
        required_columns: list of required column names
        allow_nan_ratio: maximum allowed ratio of NaN values per column
    
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'missing_columns': [],
        'high_nan_columns': [],
        'total_rows': len(data),
        'total_columns': len(data.columns)
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            validation_result['missing_columns'] = missing
            validation_result['is_valid'] = False
    
    for column in data.columns:
        nan_ratio = data[column].isna().sum() / len(data)
        if nan_ratio > allow_nan_ratio:
            validation_result['high_nan_columns'].append({
                'column': column,
                'nan_ratio': nan_ratio
            })
            validation_result['is_valid'] = False
    
    return validation_result

def example_usage():
    """
    Example usage of the data cleaning utilities.
    """
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'id': range(100),
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    print("Original data shape:", sample_data.shape)
    
    cleaned = clean_dataset(
        sample_data,
        numeric_columns=['feature_a', 'feature_b'],
        outlier_method='iqr',
        normalize_method='zscore'
    )
    
    print("Cleaned data shape:", cleaned.shape)
    
    validation = validate_data(
        cleaned,
        required_columns=['feature_a_normalized', 'feature_b_normalized']
    )
    
    print("Validation result:", validation['is_valid'])
    
    return cleaned

if __name__ == "__main__":
    result = example_usage()
    print("\nFirst 5 rows of cleaned data:")
    print(result.head())