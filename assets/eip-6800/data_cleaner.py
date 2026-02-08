
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using the Interquartile Range method.
    Returns a boolean mask where True indicates an outlier.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers(data, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    Returns a cleaned DataFrame.
    """
    outlier_mask = detect_outliers_iqr(data, column, threshold)
    return data[~outlier_mask].copy()

def normalize_minmax(data, column):
    """
    Normalize data to [0, 1] range using min-max scaling.
    Returns a Series with normalized values.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return pd.Series(0.5, index=data.index)
    return (data[column] - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using z-score standardization.
    Returns a Series with standardized values.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return pd.Series(0, index=data.index)
    return (data[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns, outlier_threshold=1.5, normalization='zscore'):
    """
    Main cleaning function: removes outliers and normalizes specified columns.
    Returns a cleaned DataFrame and a dictionary of cleaning statistics.
    """
    cleaned_df = df.copy()
    stats_dict = {
        'original_rows': len(df),
        'outliers_removed': 0,
        'cleaned_columns': []
    }
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            outlier_mask = detect_outliers_iqr(cleaned_df, col, outlier_threshold)
            outliers_count = outlier_mask.sum()
            
            if outliers_count > 0:
                cleaned_df = cleaned_df[~outlier_mask]
                stats_dict['outliers_removed'] += outliers_count
            
            if normalization == 'minmax':
                cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
            else:
                cleaned_df[f'{col}_normalized'] = normalize_zscore(cleaned_df, col)
            
            stats_dict['cleaned_columns'].append(col)
    
    stats_dict['final_rows'] = len(cleaned_df)
    stats_dict['rows_removed'] = stats_dict['original_rows'] - stats_dict['final_rows']
    
    return cleaned_df, stats_dict

def validate_data(df, required_columns, numeric_columns):
    """
    Validate dataset structure and content.
    Returns a tuple of (is_valid, error_messages).
    """
    errors = []
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    for col in numeric_columns:
        if col in df.columns:
            if not np.issubdtype(df[col].dtype, np.number):
                errors.append(f"Column '{col}' must be numeric")
            if df[col].isnull().all():
                errors.append(f"Column '{col}' contains only null values")
    
    return len(errors) == 0, errors

def example_usage():
    """
    Demonstrate the data cleaning functions with sample data.
    """
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'id': range(100),
        'feature_a': np.random.normal(50, 10, 100),
        'feature_b': np.random.exponential(5, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    print("Original data shape:", sample_data.shape)
    
    is_valid, errors = validate_data(
        sample_data, 
        required_columns=['id', 'feature_a', 'feature_b'],
        numeric_columns=['feature_a', 'feature_b']
    )
    
    if is_valid:
        cleaned_data, stats = clean_dataset(
            sample_data,
            numeric_columns=['feature_a', 'feature_b'],
            outlier_threshold=1.5,
            normalization='zscore'
        )
        
        print(f"Cleaned data shape: {cleaned_data.shape}")
        print(f"Outliers removed: {stats['outliers_removed']}")
        print(f"Rows removed: {stats['rows_removed']}")
        print(f"Cleaned columns: {stats['cleaned_columns']}")
        
        return cleaned_data
    else:
        print("Validation errors:", errors)
        return None

if __name__ == "__main__":
    result = example_usage()
    if result is not None:
        print("\nFirst 5 rows of cleaned data:")
        print(result.head())