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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df):
    """
    Calculate summary statistics for numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: Summary statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return pd.DataFrame()
    
    stats = df[numeric_cols].agg(['count', 'mean', 'std', 'min', 'max'])
    stats.loc['median'] = df[numeric_cols].median()
    stats.loc['q1'] = df[numeric_cols].quantile(0.25)
    stats.loc['q3'] = df[numeric_cols].quantile(0.75)
    
    return stats

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Imputation strategy ('mean', 'median', 'drop')
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if strategy not in ['mean', 'median', 'drop']:
        raise ValueError("Strategy must be 'mean', 'median', or 'drop'")
    
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        df_clean = df_clean.dropna(subset=numeric_cols)
    else:
        for col in numeric_cols:
            if strategy == 'mean':
                fill_value = df_clean[col].mean()
            else:
                fill_value = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(fill_value)
    
    return df_clean.reset_index(drop=True)