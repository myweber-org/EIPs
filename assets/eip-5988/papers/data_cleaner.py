
import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using IQR method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_data = data[(z_scores < threshold) | (data[column].isna())]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_data(data, column):
    """
    Standardize data using Z-score normalization
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns, outlier_method='zscore', normalize=False, standardize=False):
    """
    Main cleaning function for datasets
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        if outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        elif outlier_method == 'iqr':
            outliers = detect_outliers_iqr(cleaned_df, col)
            cleaned_df = cleaned_df.drop(outliers.index)
        
        if normalize:
            cleaned_df[col + '_normalized'] = normalize_minmax(cleaned_df, col)
        
        if standardize:
            cleaned_df[col + '_standardized'] = standardize_data(cleaned_df, col)
    
    return cleaned_df

def handle_missing_values(df, strategy='mean', fill_value=None):
    """
    Handle missing values in dataframe
    """
    df_filled = df.copy()
    
    for col in df_filled.columns:
        if df_filled[col].isna().any():
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(df_filled[col]):
                df_filled[col].fillna(df_filled[col].mean(), inplace=True)
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df_filled[col]):
                df_filled[col].fillna(df_filled[col].median(), inplace=True)
            elif strategy == 'mode':
                df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
            elif strategy == 'ffill':
                df_filled[col].fillna(method='ffill', inplace=True)
            elif strategy == 'bfill':
                df_filled[col].fillna(method='bfill', inplace=True)
            elif fill_value is not None:
                df_filled[col].fillna(fill_value, inplace=True)
            else:
                df_filled[col].fillna(0, inplace=True)
    
    return df_filled

def validate_dataframe(df, required_columns=None, numeric_check=True):
    """
    Validate dataframe structure and content
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'non_numeric_columns': [],
        'null_counts': {}
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
    
    if numeric_check:
        non_numeric = [col for col in df.select_dtypes(exclude=[np.number]).columns]
        validation_results['non_numeric_columns'] = non_numeric
    
    null_counts = df.isnull().sum()
    validation_results['null_counts'] = null_counts[null_counts > 0].to_dict()
    
    if len(validation_results['null_counts']) > 0:
        validation_results['is_valid'] = False
    
    return validation_results