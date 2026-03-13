
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, columns):
    """
    Normalize specified columns using min-max scaling
    """
    df_normalized = df.copy()
    for col in columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
    return df_normalized

def standardize_zscore(df, columns):
    """
    Standardize specified columns using z-score normalization
    """
    df_standardized = df.copy()
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df_standardized[col] = (df[col] - mean_val) / std_val
    return df_standardized

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy
    """
    df_processed = df.copy()
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if col in df.columns and df[col].isnull().any():
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            elif strategy == 'zero':
                fill_value = 0
            else:
                fill_value = df[col].mean()
            
            df_processed[col] = df[col].fillna(fill_value)
    
    return df_processed

def clean_dataset(df, config):
    """
    Main cleaning function applying multiple preprocessing steps
    """
    cleaned_df = df.copy()
    
    if 'outlier_columns' in config:
        for col in config['outlier_columns']:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    if 'missing_strategy' in config:
        cleaned_df = handle_missing_values(
            cleaned_df, 
            strategy=config['missing_strategy'],
            columns=config.get('missing_columns')
        )
    
    if 'normalize_columns' in config:
        cleaned_df = normalize_minmax(cleaned_df, config['normalize_columns'])
    
    if 'standardize_columns' in config:
        cleaned_df = standardize_zscore(cleaned_df, config['standardize_columns'])
    
    return cleaned_df