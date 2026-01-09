import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to check for missing values.
                                  If None, checks all columns.
    
    Returns:
        pd.DataFrame: DataFrame with rows containing missing values removed
    """
    if columns is None:
        return df.dropna()
    else:
        return df.dropna(subset=columns)

def fill_missing_with_mean(df, columns):
    """
    Fill missing values in specified columns with their mean.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to fill
    
    Returns:
        pd.DataFrame: DataFrame with missing values filled
    """
    df_filled = df.copy()
    for col in columns:
        if col in df_filled.columns:
            df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
    return df_filled

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a column using the IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check for outliers
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        return df
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def standardize_column(df, column):
    """
    Standardize a column using z-score normalization.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to standardize
    
    Returns:
        pd.DataFrame: DataFrame with standardized column
    """
    if column not in df.columns:
        return df
    
    df_standardized = df.copy()
    mean = df_standardized[column].mean()
    std = df_standardized[column].std()
    
    if std > 0:
        df_standardized[column] = (df_standardized[column] - mean) / std
    
    return df_standardized

def clean_dataset(df, missing_strategy='remove', outlier_columns=None):
    """
    Comprehensive dataset cleaning function.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_strategy (str): Strategy for handling missing values.
                               Options: 'remove', 'mean'
        outlier_columns (list): List of columns to remove outliers from
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'remove':
        cleaned_df = remove_missing_rows(cleaned_df)
    elif missing_strategy == 'mean':
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
        cleaned_df = fill_missing_with_mean(cleaned_df, numeric_cols)
    
    # Remove outliers
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    return cleaned_df