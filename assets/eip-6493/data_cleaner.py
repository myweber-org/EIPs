import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def clean_dataset(df, numeric_columns=None):
    """
    Clean a dataset by removing outliers from all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_dfimport pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def handle_missing_values(data, strategy='mean', columns=None):
    if columns is None:
        columns = data.columns
    
    data_filled = data.copy()
    
    for col in columns:
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'constant'")
            
            data_filled[col] = data_filled[col].fillna(fill_value)
    
    return data_filled

def remove_duplicates(data, subset=None, keep='first'):
    return data.drop_duplicates(subset=subset, keep=keep)

def normalize_data(data, columns=None, method='minmax'):
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    data_normalized = data.copy()
    
    for col in columns:
        if method == 'minmax':
            min_val = data[col].min()
            max_val = data[col].max()
            if max_val != min_val:
                data_normalized[col] = (data[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = data[col].mean()
            std_val = data[col].std()
            if std_val != 0:
                data_normalized[col] = (data[col] - mean_val) / std_val
        else:
            raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return data_normalized

def clean_dataset(data, outlier_columns=None, missing_strategy='mean', normalize_method=None):
    cleaned_data = data.copy()
    
    cleaned_data = remove_duplicates(cleaned_data)
    
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_data.columns:
                outliers = detect_outliers_iqr(cleaned_data, col)
                if not outliers.empty:
                    Q1 = cleaned_data[col].quantile(0.25)
                    Q3 = cleaned_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    cleaned_data[col] = cleaned_data[col].clip(lower=lower_bound, upper=upper_bound)
    
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    
    if normalize_method:
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        cleaned_data = normalize_data(cleaned_data, columns=numeric_cols, method=normalize_method)
    
    return cleaned_data