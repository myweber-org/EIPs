
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_minmax(df, columns):
    normalized_df = df.copy()
    for col in columns:
        min_val = normalized_df[col].min()
        max_val = normalized_df[col].max()
        normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    return normalized_df

def handle_missing_values(df, strategy='mean'):
    processed_df = df.copy()
    for col in processed_df.columns:
        if processed_df[col].isnull().any():
            if strategy == 'mean':
                fill_value = processed_df[col].mean()
            elif strategy == 'median':
                fill_value = processed_df[col].median()
            elif strategy == 'mode':
                fill_value = processed_df[col].mode()[0]
            else:
                fill_value = 0
            processed_df[col].fillna(fill_value, inplace=True)
    return processed_df

def validate_dataframe(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        raise ValueError("DataFrame is empty")
    return True

def process_dataset(file_path, outlier_cols=None, normalize_cols=None):
    try:
        df = pd.read_csv(file_path)
        validate_dataframe(df)
        
        df = handle_missing_values(df)
        
        if outlier_cols:
            df = remove_outliers_iqr(df, outlier_cols)
        
        if normalize_cols:
            df = normalize_minmax(df, normalize_cols)
        
        return df
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return None