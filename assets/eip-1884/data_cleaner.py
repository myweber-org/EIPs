
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(df, columns):
    """
    Remove outliers using IQR method
    """
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def remove_outliers_zscore(df, columns, threshold=3):
    """
    Remove outliers using Z-score method
    """
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            z_scores = np.abs(stats.zscore(df_clean[col]))
            df_clean = df_clean[z_scores < threshold]
    return df_clean

def normalize_minmax(df, columns):
    """
    Normalize data using min-max scaling
    """
    df_normalized = df.copy()
    for col in columns:
        if col in df_normalized.columns:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            if max_val != min_val:
                df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    return df_normalized

def normalize_zscore(df, columns):
    """
    Normalize data using Z-score standardization
    """
    df_normalized = df.copy()
    for col in columns:
        if col in df_normalized.columns:
            mean_val = df_normalized[col].mean()
            std_val = df_normalized[col].std()
            if std_val > 0:
                df_normalized[col] = (df_normalized[col] - mean_val) / std_val
    return df_normalized

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values with different strategies
    """
    df_filled = df.copy()
    if columns is None:
        columns = df_filled.columns
    
    for col in columns:
        if col in df_filled.columns and df_filled[col].isnull().any():
            if strategy == 'mean':
                fill_value = df_filled[col].mean()
            elif strategy == 'median':
                fill_value = df_filled[col].median()
            elif strategy == 'mode':
                fill_value = df_filled[col].mode()[0]
            elif strategy == 'zero':
                fill_value = 0
            else:
                continue
            
            df_filled[col] = df_filled[col].fillna(fill_value)
    
    return df_filled

def clean_data_pipeline(df, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    """
    Complete data cleaning pipeline
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if outlier_method == 'iqr':
        df = remove_outliers_iqr(df, numeric_cols)
    elif outlier_method == 'zscore':
        df = remove_outliers_zscore(df, numeric_cols)
    
    df = handle_missing_values(df, strategy=missing_strategy, columns=numeric_cols)
    
    if normalize_method == 'minmax':
        df = normalize_minmax(df, numeric_cols)
    elif normalize_method == 'zscore':
        df = normalize_zscore(df, numeric_cols)
    
    return df

def get_data_summary(df):
    """
    Generate summary statistics for data quality assessment
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    if summary['numeric_columns']:
        numeric_stats = df[summary['numeric_columns']].describe().to_dict()
        summary['numeric_statistics'] = numeric_stats
    
    return summarydef remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    unique_list = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list

def clean_numeric_strings(data_list):
    """
    Convert string representations of numbers to integers where possible.
    Non-convertible items remain unchanged.
    """
    cleaned = []
    for item in data_list:
        if isinstance(item, str) and item.isdigit():
            cleaned.append(int(item))
        else:
            cleaned.append(item)
    return cleaned

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, "4", "4", "five", 3, 1]
    print("Original:", sample_data)
    
    unique_data = remove_duplicates(sample_data)
    print("After removing duplicates:", unique_data)
    
    cleaned_data = clean_numeric_strings(unique_data)
    print("After cleaning numeric strings:", cleaned_data)