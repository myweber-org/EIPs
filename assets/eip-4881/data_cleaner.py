import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: multiplier for IQR (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling to range [0, 1].
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data
    
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def clean_dataset(df, numeric_columns, outlier_factor=1.5):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        df: pandas DataFrame
        numeric_columns: list of numeric column names
        outlier_factor: IQR factor for outlier detection
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_factor)
            cleaned_df = normalize_minmax(cleaned_df, col)
    
    return cleaned_df

def calculate_statistics(df, numeric_columns):
    """
    Calculate basic statistics for numeric columns.
    
    Args:
        df: pandas DataFrame
        numeric_columns: list of numeric column names
    
    Returns:
        Dictionary with statistics for each column
    """
    stats = {}
    
    for col in numeric_columns:
        if col in df.columns:
            stats[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'count': df[col].count()
            }
    
    return stats

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'temperature': [22.5, 23.1, 21.8, 100.2, 22.9, 23.5, -10.3, 22.7],
        'humidity': [45, 48, 42, 150, 47, 49, -5, 46],
        'pressure': [1013, 1012, 1014, 2000, 1013, 1012, 500, 1014]
    })
    
    print("Original data:")
    print(sample_data)
    print("\nStatistics before cleaning:")
    print(calculate_statistics(sample_data, ['temperature', 'humidity', 'pressure']))
    
    cleaned_data = clean_dataset(sample_data, ['temperature', 'humidity', 'pressure'])
    
    print("\nCleaned data:")
    print(cleaned_data)
    print("\nStatistics after cleaning:")
    print(calculate_statistics(cleaned_data, ['temperature', 'humidity', 'pressure']))