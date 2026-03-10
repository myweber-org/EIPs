
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling to [0, 1] range.
    
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

def z_score_normalize(data, column):
    """
    Normalize data using z-score standardization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with z-score normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    z_scores = (data[column] - mean_val) / std_val
    return z_scores

def detect_skewness(data, column, threshold=0.5):
    """
    Detect skewness in data distribution.
    
    Args:
        data: pandas DataFrame
        column: column name to analyze
        threshold: absolute skewness threshold for detection
    
    Returns:
        Tuple of (skewness_value, is_skewed)
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    skewness = data[column].skew()
    is_skewed = abs(skewness) > threshold
    
    return skewness, is_skewed

def clean_dataset(data, numeric_columns=None, outlier_multiplier=1.5):
    """
    Comprehensive dataset cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process
        outlier_multiplier: IQR multiplier for outlier removal
    
    Returns:
        Cleaned DataFrame and cleaning statistics
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    original_shape = data.shape
    cleaned_data = data.copy()
    stats_report = {}
    
    for col in numeric_columns:
        if col not in cleaned_data.columns:
            continue
            
        # Remove outliers
        before_count = len(cleaned_data)
        cleaned_data = remove_outliers_iqr(cleaned_data, col, outlier_multiplier)
        outliers_removed = before_count - len(cleaned_data)
        
        # Normalize
        cleaned_data[f"{col}_normalized"] = normalize_minmax(cleaned_data, col)
        
        # Calculate statistics
        skewness, is_skewed = detect_skewness(cleaned_data, col)
        
        stats_report[col] = {
            'outliers_removed': outliers_removed,
            'skewness': skewness,
            'is_skewed': is_skewed,
            'mean': cleaned_data[col].mean(),
            'std': cleaned_data[col].std()
        }
    
    final_shape = cleaned_data.shape
    stats_report['dataset'] = {
        'original_rows': original_shape[0],
        'final_rows': final_shape[0],
        'rows_removed': original_shape[0] - final_shape[0],
        'columns_added': final_shape[1] - original_shape[1]
    }
    
    return cleaned_data, stats_report

def example_usage():
    """Example demonstrating the data cleaning utilities."""
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000)
    })
    
    # Add some outliers
    sample_data.loc[1000:1005, 'feature_a'] = [500, -200, 300, 400, 600, 700]
    
    print("Original dataset shape:", sample_data.shape)
    print("\nFirst few rows:")
    print(sample_data.head())
    
    # Clean the dataset
    cleaned_data, stats = clean_dataset(sample_data)
    
    print("\nCleaned dataset shape:", cleaned_data.shape)
    print("\nCleaning statistics:")
    for col, col_stats in stats.items():
        if col != 'dataset':
            print(f"\n{col}:")
            for key, value in col_stats.items():
                print(f"  {key}: {value:.4f}")
    
    print(f"\nDataset summary:")
    for key, value in stats['dataset'].items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    example_usage()
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
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling to range [0, 1].
    
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
    Handle missing values in specified columns.
    
    Args:
        data: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: list of columns to process (None for all numeric columns)
    
    Returns:
        DataFrame with missing values handled
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    result = data.copy()
    
    for col in columns:
        if col not in result.columns:
            continue
            
        if strategy == 'drop':
            result = result.dropna(subset=[col])
        elif strategy == 'mean':
            result[col] = result[col].fillna(result[col].mean())
        elif strategy == 'median':
            result[col] = result[col].fillna(result[col].median())
        elif strategy == 'mode':
            result[col] = result[col].fillna(result[col].mode()[0])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return result

def validate_dataframe(data, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        data: pandas DataFrame to validate
        required_columns: list of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if data.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"