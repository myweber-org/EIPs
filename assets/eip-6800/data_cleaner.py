
import numpy as np
import pandas as pd

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

def normalize_column(df, column, method='minmax'):
    """
    Normalize a DataFrame column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val != 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'values': np.random.normal(100, 15, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original statistics:", calculate_basic_stats(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_basic_stats(cleaned_df, 'values'))
    
    normalized_df = normalize_column(cleaned_df, 'values', method='zscore')
    print("\nNormalized statistics:", calculate_basic_stats(normalized_df, 'values'))
    
    return cleaned_df, normalized_df

if __name__ == "__main__":
    cleaned, normalized = example_usage()
import numpy as np
import pandas as pd

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers in a pandas Series using the Interquartile Range method.
    
    Parameters:
    data (pd.Series): Input data series
    column (str): Column name to analyze
    threshold (float): Multiplier for IQR (default 1.5)
    
    Returns:
    pd.Series: Boolean mask where True indicates outliers
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = data[column]
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    return (series < lower_bound) | (series > upper_bound)

def remove_outliers(data, column, threshold=1.5, inplace=False):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    threshold (float): Multiplier for IQR (default 1.5)
    inplace (bool): Whether to modify the DataFrame in place
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    outlier_mask = detect_outliers_iqr(data, column, threshold)
    
    if inplace:
        data.drop(data[outlier_mask].index, inplace=True)
        return data
    else:
        return data[~outlier_mask].copy()

def winsorize_data(data, column, limits=(0.05, 0.05)):
    """
    Apply winsorization to limit extreme values in a column.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to winsorize
    limits (tuple): Lower and upper percentile limits
    
    Returns:
    pd.Series: Winsorized column values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    series = data[column].copy()
    lower_limit = series.quantile(limits[0])
    upper_limit = series.quantile(1 - limits[1])
    
    series[series < lower_limit] = lower_limit
    series[series > upper_limit] = upper_limit
    
    return series

def clean_dataset(data, numeric_columns=None, outlier_threshold=1.5, winsorize_limits=(0.01, 0.01)):
    """
    Comprehensive data cleaning pipeline for numeric columns.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names to clean
    outlier_threshold (float): IQR multiplier for outlier detection
    winsorize_limits (tuple): Percentile limits for winsorization
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            # Remove extreme outliers
            cleaned_data = remove_outliers(cleaned_data, column, outlier_threshold, inplace=False)
            
            # Apply winsorization to remaining data
            cleaned_data[column] = winsorize_data(cleaned_data, column, winsorize_limits)
    
    return cleaned_data.reset_index(drop=True)

def get_cleaning_report(data, cleaned_data, numeric_columns=None):
    """
    Generate a report comparing original and cleaned data.
    
    Parameters:
    data (pd.DataFrame): Original DataFrame
    cleaned_data (pd.DataFrame): Cleaned DataFrame
    numeric_columns (list): List of numeric column names to report
    
    Returns:
    pd.DataFrame: Summary statistics comparison
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    report_data = []
    
    for column in numeric_columns:
        if column in data.columns and column in cleaned_data.columns:
            original_stats = data[column].describe()
            cleaned_stats = cleaned_data[column].describe()
            
            report_data.append({
                'column': column,
                'original_count': original_stats['count'],
                'cleaned_count': cleaned_stats['count'],
                'removed_count': original_stats['count'] - cleaned_stats['count'],
                'original_mean': original_stats['mean'],
                'cleaned_mean': cleaned_stats['mean'],
                'original_std': original_stats['std'],
                'cleaned_std': cleaned_stats['std'],
                'original_min': original_stats['min'],
                'cleaned_min': cleaned_stats['min'],
                'original_max': original_stats['max'],
                'cleaned_max': cleaned_stats['max']
            })
    
    return pd.DataFrame(report_data)