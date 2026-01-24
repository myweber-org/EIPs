
import numpy as np
import pandas as pd

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
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling to range [0, 1].
    
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
    Standardize data using Z-score normalization.
    
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

def clean_dataset(data, numeric_columns, outlier_multiplier=1.5, normalization_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names to process
        outlier_multiplier: multiplier for IQR outlier detection
        normalization_method: 'minmax' or 'zscore'
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            # Remove outliers
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_multiplier)
            
            # Apply normalization
            if normalization_method == 'minmax':
                cleaned_data[f'{column}_normalized'] = normalize_minmax(cleaned_data, column)
            elif normalization_method == 'zscore':
                cleaned_data[f'{column}_standardized'] = standardize_zscore(cleaned_data, column)
            else:
                raise ValueError("normalization_method must be 'minmax' or 'zscore'")
    
    return cleaned_data

def validate_data(data, required_columns, numeric_threshold=0.8):
    """
    Validate dataset structure and content.
    
    Args:
        data: pandas DataFrame to validate
        required_columns: list of required column names
        numeric_threshold: minimum proportion of numeric values in numeric columns
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'has_required_columns': True,
        'missing_columns': [],
        'numeric_quality': {},
        'row_count': len(data),
        'column_count': len(data.columns)
    }
    
    # Check required columns
    for column in required_columns:
        if column not in data.columns:
            validation_results['has_required_columns'] = False
            validation_results['missing_columns'].append(column)
    
    # Check numeric columns quality
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for column in numeric_cols:
        non_null_count = data[column].notna().sum()
        numeric_ratio = non_null_count / len(data)
        validation_results['numeric_quality'][column] = {
            'non_null_count': non_null_count,
            'numeric_ratio': numeric_ratio,
            'is_valid': numeric_ratio >= numeric_threshold
        }
    
    return validation_results

# Example usage demonstration
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'id': range(100),
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Add some outliers
    sample_data.loc[10, 'feature_a'] = 500
    sample_data.loc[20, 'feature_b'] = 1000
    
    print("Original data shape:", sample_data.shape)
    print("\nData validation:")
    validation = validate_data(sample_data, ['id', 'feature_a', 'feature_b'])
    print(validation)
    
    print("\nCleaning data...")
    cleaned = clean_dataset(sample_data, ['feature_a', 'feature_b'], normalization_method='zscore')
    print("Cleaned data shape:", cleaned.shape)
    print("\nCleaned data summary:")
    print(cleaned[['feature_a_standardized', 'feature_b_standardized']].describe())