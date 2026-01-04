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
    return filtered_data

def z_score_normalize(data, column):
    """
    Normalize data using z-score method
    """
    mean = data[column].mean()
    std = data[column].std()
    
    if std == 0:
        return data[column]
    
    normalized = (data[column] - mean) / std
    return normalized

def min_max_normalize(data, column):
    """
    Normalize data using min-max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column]
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='zscore'):
    """
    Main cleaning function for datasets
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            z_scores = np.abs(stats.zscore(cleaned_df[col]))
            cleaned_df = cleaned_df[z_scores < 3]
        
        if normalize_method == 'zscore':
            cleaned_df[f'{col}_normalized'] = z_score_normalize(cleaned_df, col)
        elif normalize_method == 'minmax':
            cleaned_df[f'{col}_normalized'] = min_max_normalize(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns, min_rows=10):
    """
    Validate dataset structure and content
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if len(df) < min_rows:
        raise ValueError(f"Dataset has fewer than {min_rows} rows")
    
    return True