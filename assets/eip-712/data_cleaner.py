
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 5, 1, 6]
    cleaned = remove_duplicates_preserve_order(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Columns to consider for duplicates
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): 'mean', 'median', 'mode', or 'drop'
        columns (list, optional): Specific columns to process
    
    Returns:
        pd.DataFrame: DataFrame with handled missing values
    """
    if columns is None:
        columns = df.columns
    
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns:
            if strategy == 'drop':
                df_copy = df_copy.dropna(subset=[col])
            elif strategy == 'mean':
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif strategy == 'median':
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif strategy == 'mode':
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
    
    return df_copy

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column in DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column to normalize
        method (str): 'minmax' or 'zscore'
    
    Returns:
        pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        return df
    
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
    
    return df_copy

def filter_outliers(df, column, method='iqr', threshold=1.5):
    """
    Filter outliers from DataFrame based on column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column to check for outliers
        method (str): 'iqr' or 'zscore'
        threshold (float): Threshold value for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        return df
    
    df_copy = df.copy()
    
    if method == 'iqr':
        Q1 = df_copy[column].quantile(0.25)
        Q3 = df_copy[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df_copy = df_copy[(df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)]
    
    elif method == 'zscore':
        z_scores = np.abs((df_copy[column] - df_copy[column].mean()) / df_copy[column].std())
        df_copy = df_copy[z_scores < threshold]
    
    return df_copy

def clean_dataframe(df, config):
    """
    Apply multiple cleaning operations based on configuration.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        config (dict): Configuration dictionary with cleaning operations
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_copy = df.copy()
    
    if 'remove_duplicates' in config:
        df_copy = remove_duplicates(df_copy, config.get('remove_duplicates'))
    
    if 'handle_missing' in config:
        missing_config = config['handle_missing']
        df_copy = handle_missing_values(
            df_copy,
            strategy=missing_config.get('strategy', 'mean'),
            columns=missing_config.get('columns')
        )
    
    if 'normalize' in config:
        for norm_config in config['normalize']:
            df_copy = normalize_column(
                df_copy,
                column=norm_config['column'],
                method=norm_config.get('method', 'minmax')
            )
    
    if 'filter_outliers' in config:
        for outlier_config in config['filter_outliers']:
            df_copy = filter_outliers(
                df_copy,
                column=outlier_config['column'],
                method=outlier_config.get('method', 'iqr'),
                threshold=outlier_config.get('threshold', 1.5)
            )
    
    return df_copy