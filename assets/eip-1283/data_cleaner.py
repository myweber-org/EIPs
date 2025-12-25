
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
    return data[mask]

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
    
    return (data[column] - min_val) / (max_val - min_val)

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
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    return (data[column] - mean_val) / std_val

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric columns)
        outlier_method: 'iqr', 'zscore', or None
        normalize_method: 'minmax', 'zscore', or None
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    # Remove outliers
    if outlier_method:
        for col in numeric_columns:
            if col in cleaned_data.columns:
                if outlier_method == 'iqr':
                    cleaned_data = remove_outliers_iqr(cleaned_data, col)
                elif outlier_method == 'zscore':
                    cleaned_data = remove_outliers_zscore(cleaned_data, col)
    
    # Normalize data
    if normalize_method:
        for col in numeric_columns:
            if col in cleaned_data.columns:
                if normalize_method == 'minmax':
                    cleaned_data[col] = normalize_minmax(cleaned_data, col)
                elif normalize_method == 'zscore':
                    cleaned_data[col] = normalize_zscore(cleaned_data, col)
    
    return cleaned_data

def validate_data(data, required_columns=None, check_missing=True, check_duplicates=True):
    """
    Validate data quality.
    
    Args:
        data: pandas DataFrame
        required_columns: list of required columns
        check_missing: flag to check for missing values
        check_duplicates: flag to check for duplicate rows
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {}
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        validation_results['missing_columns'] = missing_columns
    
    # Check missing values
    if check_missing:
        missing_counts = data.isnull().sum()
        missing_percentage = (missing_counts / len(data)) * 100
        validation_results['missing_values'] = missing_counts[missing_counts > 0].to_dict()
        validation_results['missing_percentage'] = missing_percentage[missing_counts > 0].to_dict()
    
    # Check duplicates
    if check_duplicates:
        duplicate_count = data.duplicated().sum()
        validation_results['duplicate_rows'] = duplicate_count
    
    # Data types
    validation_results['dtypes'] = data.dtypes.to_dict()
    
    # Basic statistics
    if len(data) > 0:
        validation_results['row_count'] = len(data)
        validation_results['column_count'] = len(data.columns)
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    # Add some outliers
    sample_data.loc[10, 'feature_a'] = 500
    sample_data.loc[20, 'feature_b'] = 1000
    
    print("Original data shape:", sample_data.shape)
    
    # Validate data
    validation = validate_data(sample_data)
    print("\nValidation results:")
    for key, value in validation.items():
        print(f"{key}: {value}")
    
    # Clean data
    cleaned = clean_dataset(
        sample_data,
        numeric_columns=['feature_a', 'feature_b', 'feature_c'],
        outlier_method='iqr',
        normalize_method='minmax'
    )
    
    print("\nCleaned data shape:", cleaned.shape)
    print("\nCleaned data summary:")
    print(cleaned.describe())