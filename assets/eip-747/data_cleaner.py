
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using IQR method.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_stats(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if data.empty:
        return {}
    
    stats = {
        'mean': np.mean(data[column]),
        'median': np.median(data[column]),
        'std': np.std(data[column]),
        'min': np.min(data[column]),
        'max': np.max(data[column]),
        'count': len(data[column])
    }
    return statsimport numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns=None, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def remove_outliers_zscore(df, columns=None, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    for col in columns:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        mask = z_scores < threshold
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(df, columns=None):
    """
    Normalize data using Min-Max scaling
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:
            df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
    
    return df_normalized

def normalize_zscore(df, columns=None):
    """
    Normalize data using Z-score standardization
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    for col in columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val > 0:
            df_normalized[col] = (df[col] - mean_val) / std_val
    
    return df_normalized

def clean_dataset(df, outlier_method='iqr', normalize_method='minmax', 
                  outlier_params=None, normalize_params=None):
    """
    Complete data cleaning pipeline
    """
    if outlier_params is None:
        outlier_params = {}
    if normalize_params is None:
        normalize_params = {}
    
    # Remove outliers
    if outlier_method == 'iqr':
        df_clean = remove_outliers_iqr(df, **outlier_params)
    elif outlier_method == 'zscore':
        df_clean = remove_outliers_zscore(df, **outlier_params)
    else:
        df_clean = df.copy()
    
    # Normalize data
    if normalize_method == 'minmax':
        df_final = normalize_minmax(df_clean, **normalize_params)
    elif normalize_method == 'zscore':
        df_final = normalize_zscore(df_clean, **normalize_params)
    else:
        df_final = df_clean
    
    return df_final

def get_cleaning_summary(original_df, cleaned_df):
    """
    Generate summary of cleaning operations
    """
    summary = {
        'original_rows': len(original_df),
        'cleaned_rows': len(cleaned_df),
        'rows_removed': len(original_df) - len(cleaned_df),
        'removal_percentage': ((len(original_df) - len(cleaned_df)) / len(original_df)) * 100
    }
    
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary[f'{col}_original_mean'] = original_df[col].mean()
        summary[f'{col}_cleaned_mean'] = cleaned_df[col].mean()
        summary[f'{col}_original_std'] = original_df[col].std()
        summary[f'{col}_cleaned_std'] = cleaned_df[col].std()
    
    return summary