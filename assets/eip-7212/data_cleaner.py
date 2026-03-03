
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to process
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    return filtered_df

def remove_outliers_zscore(dataframe, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to process
        threshold: Z-score threshold for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(dataframe[column].dropna()))
    filtered_indices = np.where(z_scores < threshold)[0]
    
    column_indices = dataframe[column].dropna().index
    valid_indices = column_indices[filtered_indices]
    
    filtered_df = dataframe.loc[valid_indices]
    return filtered_df

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using Min-Max scaling.
    
    Args:
        dataframe: pandas DataFrame
        columns: List of columns to normalize. If None, normalize all numeric columns.
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[col]):
            col_min = dataframe[col].min()
            col_max = dataframe[col].max()
            
            if col_max != col_min:
                normalized_df[col] = (dataframe[col] - col_min) / (col_max - col_min)
            else:
                normalized_df[col] = 0
    
    return normalized_df

def normalize_zscore(dataframe, columns=None):
    """
    Normalize specified columns using Z-score normalization.
    
    Args:
        dataframe: pandas DataFrame
        columns: List of columns to normalize. If None, normalize all numeric columns.
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[col]):
            col_mean = dataframe[col].mean()
            col_std = dataframe[col].std()
            
            if col_std != 0:
                normalized_df[col] = (dataframe[col] - col_mean) / col_std
            else:
                normalized_df[col] = 0
    
    return normalized_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Args:
        dataframe: pandas DataFrame
        strategy: Imputation strategy ('mean', 'median', 'mode', 'drop')
        columns: List of columns to process. If None, process all columns.
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = dataframe.columns
    
    processed_df = dataframe.copy()
    
    for col in columns:
        if col not in processed_df.columns:
            continue
            
        if processed_df[col].isnull().any():
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(processed_df[col]):
                fill_value = processed_df[col].mean()
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(processed_df[col]):
                fill_value = processed_df[col].median()
            elif strategy == 'mode':
                fill_value = processed_df[col].mode()[0] if not processed_df[col].mode().empty else None
            elif strategy == 'drop':
                processed_df = processed_df.dropna(subset=[col])
                continue
            else:
                fill_value = None
            
            if fill_value is not None:
                processed_df[col] = processed_df[col].fillna(fill_value)
    
    return processed_df

def clean_dataset(dataframe, outlier_method='iqr', normalization_method='minmax', 
                  missing_strategy='mean', outlier_columns=None, normalize_columns=None):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        dataframe: Input pandas DataFrame
        outlier_method: 'iqr', 'zscore', or None
        normalization_method: 'minmax', 'zscore', or None
        missing_strategy: 'mean', 'median', 'mode', or 'drop'
        outlier_columns: Columns for outlier removal
        normalize_columns: Columns for normalization
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = dataframe.copy()
    
    if missing_strategy:
        cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if outlier_method and outlier_columns:
        for col in outlier_columns:
            if col in cleaned_df.columns:
                if outlier_method == 'iqr':
                    cleaned_df = remove_outliers_iqr(cleaned_df, col)
                elif outlier_method == 'zscore':
                    cleaned_df = remove_outliers_zscore(cleaned_df, col)
    
    if normalization_method and normalize_columns:
        if normalization_method == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, normalize_columns)
        elif normalization_method == 'zscore':
            cleaned_df = normalize_zscore(cleaned_df, normalize_columns)
    
    return cleaned_df