
def remove_duplicates_preserve_order(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport numpy as np
import pandas as pd

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_minmax(df, columns):
    """
    Normalize specified columns using Min-Max scaling.
    """
    normalized_df = df.copy()
    for col in columns:
        if col in normalized_df.columns:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val != min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
            else:
                normalized_df[col] = 0
    return normalized_df

def standardize_zscore(df, columns):
    """
    Standardize specified columns using Z-score normalization.
    """
    standardized_df = df.copy()
    for col in columns:
        if col in standardized_df.columns:
            mean_val = standardized_df[col].mean()
            std_val = standardized_df[col].std()
            if std_val > 0:
                standardized_df[col] = (standardized_df[col] - mean_val) / std_val
            else:
                standardized_df[col] = 0
    return standardized_df

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    """
    handled_df = df.copy()
    if columns is None:
        columns = handled_df.columns
    
    for col in columns:
        if col in handled_df.columns and handled_df[col].isnull().any():
            if strategy == 'mean':
                fill_value = handled_df[col].mean()
            elif strategy == 'median':
                fill_value = handled_df[col].median()
            elif strategy == 'mode':
                fill_value = handled_df[col].mode()[0]
            elif strategy == 'drop':
                handled_df = handled_df.dropna(subset=[col])
                continue
            else:
                fill_value = 0
            
            handled_df[col] = handled_df[col].fillna(fill_value)
    
    return handled_df

def clean_dataset(df, numeric_columns, outlier_factor=1.5, normalize=True, standardize=False, missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    """
    df_clean = df.copy()
    
    df_clean = handle_missing_values(df_clean, strategy=missing_strategy, columns=numeric_columns)
    df_clean = remove_outliers_iqr(df_clean, numeric_columns, factor=outlier_factor)
    
    if normalize:
        df_clean = normalize_minmax(df_clean, numeric_columns)
    
    if standardize:
        df_clean = standardize_zscore(df_clean, numeric_columns)
    
    return df_clean