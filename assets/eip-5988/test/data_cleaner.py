
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column, threshold=1.5):
    """
    Remove outliers using IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, columns):
    """
    Normalize specified columns using min-max scaling.
    """
    df_normalized = df.copy()
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val - min_val != 0:
            df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df_normalized[col] = 0
    return df_normalized

def clean_dataset(df, numeric_columns, outlier_threshold=1.5):
    """
    Full cleaning pipeline: remove outliers and normalize numeric columns.
    """
    df_clean = df.copy()
    for col in numeric_columns:
        df_clean = remove_outliers_iqr(df_clean, col, outlier_threshold)
    df_clean = normalize_minmax(df_clean, numeric_columns)
    return df_clean.reset_index(drop=True)

def validate_data(df, required_columns):
    """
    Validate that required columns exist and have no null values.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Null values found in columns: {null_counts[null_counts > 0].to_dict()}")
    
    return True