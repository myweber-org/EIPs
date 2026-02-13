
import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: Column name to process
        multiplier: IQR multiplier (default 1.5)
    
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
    
    return filtered_df.copy()

def normalize_minmax(dataframe, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Args:
        dataframe: pandas DataFrame
        columns: List of columns to normalize (default: all numeric columns)
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    result_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and np.issubdtype(dataframe[col].dtype, np.number):
            col_min = dataframe[col].min()
            col_max = dataframe[col].max()
            
            if col_max != col_min:
                result_df[col] = (dataframe[col] - col_min) / (col_max - col_min)
            else:
                result_df[col] = 0
    
    return result_df

def standardize_zscore(dataframe, columns=None):
    """
    Standardize specified columns using z-score normalization.
    
    Args:
        dataframe: pandas DataFrame
        columns: List of columns to standardize (default: all numeric columns)
    
    Returns:
        DataFrame with standardized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    result_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and np.issubdtype(dataframe[col].dtype, np.number):
            col_mean = dataframe[col].mean()
            col_std = dataframe[col].std()
            
            if col_std > 0:
                result_df[col] = (dataframe[col] - col_mean) / col_std
            else:
                result_df[col] = 0
    
    return result_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Args:
        dataframe: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: List of columns to process (default: all columns)
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = dataframe.columns.tolist()
    
    result_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and dataframe[col].isnull().any():
            if strategy == 'mean' and np.issubdtype(dataframe[col].dtype, np.number):
                result_df[col].fillna(dataframe[col].mean(), inplace=True)
            elif strategy == 'median' and np.issubdtype(dataframe[col].dtype, np.number):
                result_df[col].fillna(dataframe[col].median(), inplace=True)
            elif strategy == 'mode':
                result_df[col].fillna(dataframe[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                result_df = result_df.dropna(subset=[col])
    
    return result_df

def create_data_summary(dataframe):
    """
    Create a summary statistics DataFrame.
    
    Args:
        dataframe: pandas DataFrame
    
    Returns:
        DataFrame with summary statistics
    """
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    
    summary_data = []
    for col in numeric_cols:
        summary_data.append({
            'column': col,
            'mean': dataframe[col].mean(),
            'median': dataframe[col].median(),
            'std': dataframe[col].std(),
            'min': dataframe[col].min(),
            'max': dataframe[col].max(),
            'missing': dataframe[col].isnull().sum(),
            'missing_percent': (dataframe[col].isnull().sum() / len(dataframe)) * 100
        })
    
    return pd.DataFrame(summary_data)