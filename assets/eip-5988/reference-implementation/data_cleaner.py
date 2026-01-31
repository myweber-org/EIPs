
import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from specified columns or entire DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of columns to check for missing values.
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
    Fill missing values in specified columns with column mean.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to fill missing values
    
    Returns:
        pd.DataFrame: DataFrame with missing values filled
    """
    df_filled = df.copy()
    for col in columns:
        if col in df_filled.columns:
            df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
    return df_filled

def detect_outliers_iqr(df, column, multiplier=1.5):
    """
    Detect outliers using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check for outliers
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame containing only the outlier rows
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to remove outliers from
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

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
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_standardized = df.copy()
    mean_val = df_standardized[column].mean()
    std_val = df_standardized[column].std()
    
    if std_val > 0:
        df_standardized[f'{column}_standardized'] = (df_standardized[column] - mean_val) / std_val
    
    return df_standardized

def clean_dataframe(df, missing_strategy='remove', outlier_strategy='remove', columns_to_clean=None):
    """
    Comprehensive data cleaning function.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_strategy (str): Strategy for handling missing values.
                               Options: 'remove', 'mean', 'ignore'
        outlier_strategy (str): Strategy for handling outliers.
                               Options: 'remove', 'detect', 'ignore'
        columns_to_clean (list, optional): Specific columns to clean.
                                          If None, cleans all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if columns_to_clean is None:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_clean = numeric_cols
    
    # Handle missing values
    if missing_strategy == 'remove':
        cleaned_df = remove_missing_rows(cleaned_df, columns_to_clean)
    elif missing_strategy == 'mean':
        cleaned_df = fill_missing_with_mean(cleaned_df, columns_to_clean)
    
    # Handle outliers
    if outlier_strategy == 'remove':
        for col in columns_to_clean:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    return cleaned_df