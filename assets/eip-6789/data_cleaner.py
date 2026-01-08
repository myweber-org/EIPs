import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using z-score standardization
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def clean_dataset(df, numeric_columns, outlier_factor=1.5, normalization_method='zscore'):
    """
    Main cleaning function for datasets
    """
    cleaned_df = df.copy()
    removal_stats = {}
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            # Remove outliers
            cleaned_df, removed = remove_outliers_iqr(cleaned_df, col, outlier_factor)
            removal_stats[col] = removed
            
            # Apply normalization
            if normalization_method == 'minmax':
                cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
            elif normalization_method == 'zscore':
                cleaned_df[f'{col}_normalized'] = normalize_zscore(cleaned_df, col)
    
    return cleaned_df, removal_stats

def validate_data(df, required_columns, numeric_threshold=0.8):
    """
    Validate dataset structure and content
    """
    validation_results = {
        'missing_columns': [],
        'low_numeric_ratio': [],
        'all_passed': True
    }
    
    # Check required columns
    for col in required_columns:
        if col not in df.columns:
            validation_results['missing_columns'].append(col)
            validation_results['all_passed'] = False
    
    # Check numeric content ratio
    for col in df.select_dtypes(include=[np.number]).columns:
        non_null_count = df[col].count()
        total_count = len(df)
        
        if total_count > 0 and (non_null_count / total_count) < numeric_threshold:
            validation_results['low_numeric_ratio'].append(col)
            validation_results['all_passed'] = False
    
    return validation_results

# Example usage function
def process_example_data():
    """
    Demonstrate the cleaning functions with sample data
    """
    np.random.seed(42)
    
    # Create sample data with outliers
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    # Add some outliers
    sample_data.loc[50:55, 'feature_a'] = 500
    sample_data.loc[200:205, 'feature_b'] = 1000
    
    print(f"Original data shape: {sample_data.shape}")
    
    # Clean the data
    numeric_cols = ['feature_a', 'feature_b']
    cleaned_data, stats = clean_dataset(sample_data, numeric_cols)
    
    print(f"Cleaned data shape: {cleaned_data.shape}")
    print(f"Outliers removed: {stats}")
    
    # Validate
    validation = validate_data(cleaned_data, numeric_cols)
    print(f"Validation passed: {validation['all_passed']}")
    
    return cleaned_data