import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas Series using the IQR method.
    Returns a filtered Series.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    Returns a filtered DataFrame.
    """
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data to [0, 1] range using min-max scaling.
    Returns a new Series.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    return (data[column] - min_val) / (max_val - min_val)

def normalize_zscore(data, column):
    """
    Normalize data using Z-score normalization (mean=0, std=1).
    Returns a new Series.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    return (data[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method=None):
    """
    Main cleaning function that applies outlier removal and optional normalization.
    Returns a cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df[col] = normalize_zscore(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def describe_cleaned_data(df, original_df, numeric_columns):
    """
    Generate comparison statistics before and after cleaning.
    Returns a summary DataFrame.
    """
    summary_data = []
    
    for col in numeric_columns:
        if col not in df.columns or col not in original_df.columns:
            continue
            
        original_stats = {
            'column': col,
            'original_count': original_df[col].count(),
            'original_mean': original_df[col].mean(),
            'original_std': original_df[col].std(),
            'cleaned_count': df[col].count(),
            'cleaned_mean': df[col].mean(),
            'cleaned_std': df[col].std(),
            'removed_percentage': ((original_df[col].count() - df[col].count()) / original_df[col].count()) * 100
        }
        summary_data.append(original_stats)
    
    return pd.DataFrame(summary_data)