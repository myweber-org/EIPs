
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
    
    return filtered_df

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, columns_to_clean=None):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of summary statistics for each cleaned column
    """
    if columns_to_clean is None:
        columns_to_clean = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    stats_summary = {}
    
    for column in columns_to_clean:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_stats(cleaned_df, column)
            stats['outliers_removed'] = removed_count
            stats_summary[column] = stats
    
    return cleaned_df, stats_summary
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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistics
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
    Normalize a column using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df = df.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df[f'{column}_normalized'] = (df[column] - min_val) / (max_val - min_val)
        else:
            df[f'{column}_normalized'] = 0
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val != 0:
            df[f'{column}_normalized'] = (df[column] - mean_val) / std_val
        else:
            df[f'{column}_normalized'] = 0
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df

def handle_missing_values(df, column, strategy='mean'):
    """
    Handle missing values in a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df = df.copy()
    
    if strategy == 'mean':
        fill_value = df[column].mean()
    elif strategy == 'median':
        fill_value = df[column].median()
    elif strategy == 'mode':
        fill_value = df[column].mode()[0] if not df[column].mode().empty else np.nan
    elif strategy == 'drop':
        return df.dropna(subset=[column])
    else:
        raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
    
    df[column] = df[column].fillna(fill_value)
    return df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'values': [1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10, np.nan, 12, 13, 14, 15]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nOriginal Statistics:")
    print(calculate_basic_stats(df, 'values'))
    
    # Handle missing values
    df_clean = handle_missing_values(df, 'values', strategy='mean')
    print("\nAfter handling missing values:")
    print(df_clean)
    
    # Remove outliers
    df_no_outliers = remove_outliers_iqr(df_clean, 'values')
    print("\nAfter removing outliers:")
    print(df_no_outliers)
    print("\nStatistics after outlier removal:")
    print(calculate_basic_stats(df_no_outliers, 'values'))
    
    # Normalize
    df_normalized = normalize_column(df_no_outliers, 'values', method='minmax')
    print("\nAfter normalization:")
    print(df_normalized)import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (list or array-like): The dataset containing the column to clean.
    column (int or str): The index or name of the column to process.
    
    Returns:
    tuple: (cleaned_data, removed_indices)
    """
    if isinstance(data, list):
        data = np.array(data)
    
    column_data = data[:, column] if data.ndim > 1 else data
    
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (column_data >= lower_bound) & (column_data <= upper_bound)
    removed_indices = np.where(~mask)[0]
    
    if data.ndim > 1:
        cleaned_data = data[mask]
    else:
        cleaned_data = data[mask]
    
    return cleaned_data, removed_indices

def validate_data_range(data, min_val=None, max_val=None):
    """
    Validate that all values in data are within specified range.
    
    Parameters:
    data (array-like): Data to validate.
    min_val (float): Minimum allowed value.
    max_val (float): Maximum allowed value.
    
    Returns:
    bool: True if all values are within range, False otherwise.
    """
    data_array = np.array(data)
    
    if min_val is not None:
        if np.any(data_array < min_val):
            return False
    
    if max_val is not None:
        if np.any(data_array > max_val):
            return False
    
    return True

def example_usage():
    # Example dataset with outliers
    sample_data = np.array([
        [1, 150],
        [2, 160],
        [3, 155],
        [4, 1000],  # Outlier
        [5, 158],
        [6, 162],
        [7, -50],   # Outlier
        [8, 165]
    ])
    
    print("Original data:")
    print(sample_data)
    
    cleaned_data, removed = remove_outliers_iqr(sample_data, column=1)
    
    print(f"\nRemoved indices: {removed}")
    print("\nCleaned data:")
    print(cleaned_data)
    
    # Validate the cleaned data
    is_valid = validate_data_range(cleaned_data[:, 1], min_val=0, max_val=200)
    print(f"\nData validation: {'Pass' if is_valid else 'Fail'}")

if __name__ == "__main__":
    example_usage()