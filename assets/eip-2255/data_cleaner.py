import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from specified columns or entire DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of columns to check. If None, checks all columns.
    
    Returns:
        pd.DataFrame: DataFrame with rows containing missing values removed
    """
    if columns is None:
        return df.dropna()
    else:
        return df.dropna(subset=columns)

def fill_missing_with_mean(df, columns):
    """
    Fill missing values in specified columns with column mean.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to fill
    
    Returns:
        pd.DataFrame: DataFrame with missing values filled
    """
    df_filled = df.copy()
    for col in columns:
        if col in df.columns:
            df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
    return df_filled

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check for outliers
        threshold (float): IQR multiplier threshold
    
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def remove_outliers(df, column, threshold=1.5):
    """
    Remove outliers from specified column using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to remove outliers from
        threshold (float): IQR multiplier threshold
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    outlier_mask = detect_outliers_iqr(df, column, threshold)
    return df[~outlier_mask].reset_index(drop=True)

def standardize_column(df, column):
    """
    Standardize a column to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to standardize
    
    Returns:
        pd.DataFrame: DataFrame with standardized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_standardized = df.copy()
    mean_val = df_standardized[column].mean()
    std_val = df_standardized[column].std()
    
    if std_val > 0:
        df_standardized[column] = (df_standardized[column] - mean_val) / std_val
    
    return df_standardized

def clean_dataset(df, missing_strategy='remove', outlier_columns=None, standardize_columns=None):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_strategy (str): Strategy for handling missing values ('remove' or 'mean')
        outlier_columns (list, optional): Columns to remove outliers from
        standardize_columns (list, optional): Columns to standardize
    
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
                cleaned_df = remove_outliers(cleaned_df, col)
    
    # Standardize columns
    if standardize_columns:
        for col in standardize_columns:
            if col in cleaned_df.columns:
                cleaned_df = standardize_column(cleaned_df, col)
    
    return cleaned_df