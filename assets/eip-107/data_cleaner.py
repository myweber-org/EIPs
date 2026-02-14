
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_summary_statistics(df):
    """
    Calculate summary statistics for numerical columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: Summary statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return pd.DataFrame()
    
    summary = df[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max'])
    
    return summary

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in numerical columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy to handle missing values ('mean', 'median', 'drop')
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        df_clean = df_clean.dropna(subset=numeric_cols)
    elif strategy == 'mean':
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    elif strategy == 'median':
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    else:
        raise ValueError("Strategy must be 'mean', 'median', or 'drop'")
    
    return df_clean