
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    Returns boolean mask for outliers.
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def remove_outliers(data, columns, threshold=1.5):
    """
    Remove outliers from specified columns using IQR method.
    """
    clean_data = data.copy()
    for col in columns:
        outliers = detect_outliers_iqr(clean_data, col, threshold)
        clean_data = clean_data[~outliers]
    return clean_data.reset_index(drop=True)

def normalize_minmax(data, columns):
    """
    Apply min-max normalization to specified columns.
    """
    normalized_data = data.copy()
    for col in columns:
        min_val = normalized_data[col].min()
        max_val = normalized_data[col].max()
        if max_val > min_val:
            normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
    return normalized_data

def standardize_zscore(data, columns):
    """
    Apply z-score standardization to specified columns.
    """
    standardized_data = data.copy()
    for col in columns:
        mean_val = standardized_data[col].mean()
        std_val = standardized_data[col].std()
        if std_val > 0:
            standardized_data[col] = (standardized_data[col] - mean_val) / std_val
    return standardized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    """
    if columns is None:
        columns = data.columns
    
    filled_data = data.copy()
    
    for col in columns:
        if filled_data[col].isnull().any():
            if strategy == 'mean':
                fill_value = filled_data[col].mean()
            elif strategy == 'median':
                fill_value = filled_data[col].median()
            elif strategy == 'mode':
                fill_value = filled_data[col].mode()[0]
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            filled_data[col] = filled_data[col].fillna(fill_value)
    
    return filled_data

def clean_dataset(data, numerical_cols, outlier_threshold=1.5, 
                  normalize=True, standardize=False, missing_strategy='mean'):
    """
    Complete data cleaning pipeline.
    """
    # Handle missing values
    cleaned_data = handle_missing_values(data, strategy=missing_strategy, columns=numerical_cols)
    
    # Remove outliers
    cleaned_data = remove_outliers(cleaned_data, numerical_cols, threshold=outlier_threshold)
    
    # Apply normalization if requested
    if normalize:
        cleaned_data = normalize_minmax(cleaned_data, numerical_cols)
    
    # Apply standardization if requested (overrides normalization for those columns)
    if standardize:
        cleaned_data = standardize_zscore(cleaned_data, numerical_cols)
    
    return cleaned_data

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 100),
        'feature2': np.random.exponential(50, 100),
        'feature3': np.random.uniform(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Add some missing values
    sample_data.loc[10:15, 'feature1'] = np.nan
    sample_data.loc[20:25, 'feature2'] = np.nan
    
    # Clean the dataset
    numerical_columns = ['feature1', 'feature2', 'feature3']
    cleaned = clean_dataset(
        sample_data, 
        numerical_cols=numerical_columns,
        outlier_threshold=1.5,
        normalize=True,
        standardize=False,
        missing_strategy='mean'
    )
    
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Missing values after cleaning: {cleaned[numerical_columns].isnull().sum().sum()}")