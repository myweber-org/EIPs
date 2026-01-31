import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def z_score_normalize(data, column):
    """
    Normalize data using z-score normalization
    """
    mean = data[column].mean()
    std = data[column].std()
    
    normalized_data = data.copy()
    normalized_data[column] = (data[column] - mean) / std
    return normalized_data

def min_max_normalize(data, column):
    """
    Normalize data using min-max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    
    normalized_data = data.copy()
    normalized_data[column] = (data[column] - min_val) / (max_val - min_val)
    return normalized_data

def clean_dataset(df, numeric_columns, outlier_factor=1.5, normalization_method='zscore'):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, column, outlier_factor)
            
            if normalization_method == 'zscore':
                cleaned_df = z_score_normalize(cleaned_df, column)
            elif normalization_method == 'minmax':
                cleaned_df = min_max_normalize(cleaned_df, column)
    
    return cleaned_df

def validate_data(df, required_columns):
    """
    Validate dataset structure and content
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if df.empty:
        raise ValueError("Dataset is empty")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    validation_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': numeric_cols,
        'has_missing_values': df.isnull().any().any(),
        'missing_values_count': df.isnull().sum().sum()
    }
    
    return validation_report