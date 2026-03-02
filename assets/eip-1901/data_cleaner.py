import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column):
    """
    Remove outliers using the Interquartile Range method.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data to range [0, 1] using min-max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns):
    """
    Apply outlier removal and normalization to specified numeric columns.
    """
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col + '_normalized'] = normalize_minmax(cleaned_df, col)
            cleaned_df[col + '_standardized'] = standardize_zscore(cleaned_df, col)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.randn(100) * 10 + 50,
        'feature_b': np.random.randn(100) * 5 + 20,
        'category': np.random.choice(['X', 'Y', 'Z'], 100)
    })
    numeric_cols = ['feature_a', 'feature_b']
    result = clean_dataset(sample_data, numeric_cols)
    print(result.head())
    print(f"Original shape: {sample_data.shape}, Cleaned shape: {result.shape}")
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (bool): Whether to fill missing values.
    strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'constant').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype in [np.float64, np.int64]:
                if strategy == 'mean':
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                elif strategy == 'median':
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                elif strategy == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
                elif strategy == 'constant':
                    cleaned_df[column].fillna(0, inplace=True)
            else:
                cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
    
    return cleaned_df

def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Remove outliers from specified columns using IQR or Z-score method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to process. If None, process all numeric columns.
    method (str): Method for outlier detection ('iqr' or 'zscore').
    threshold (float): Threshold for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for column in columns:
        if method == 'iqr':
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
        elif method == 'zscore':
            z_scores = np.abs((df_clean[column] - df_clean[column].mean()) / df_clean[column].std())
            df_clean = df_clean[z_scores < threshold]
    
    return df_clean

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize specified columns in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to normalize. If None, normalize all numeric columns.
    method (str): Normalization method ('minmax' or 'standard').
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    
    for column in columns:
        if method == 'minmax':
            min_val = df_normalized[column].min()
            max_val = df_normalized[column].max()
            if max_val != min_val:
                df_normalized[column] = (df_normalized[column] - min_val) / (max_val - min_val)
        elif method == 'standard':
            mean_val = df_normalized[column].mean()
            std_val = df_normalized[column].std()
            if std_val != 0:
                df_normalized[column] = (df_normalized[column] - mean_val) / std_val
    
    return df_normalized

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, np.nan, 7, 8, 9, 100],
        'B': [10, 20, 20, 40, 50, 60, 70, 80, 90, 1000],
        'C': ['a', 'b', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataset(df, strategy='mean')
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    outlier_removed = remove_outliers(cleaned_df, method='iqr')
    print("DataFrame after outlier removal:")
    print(outlier_removed)
    print("\n")
    
    normalized_df = normalize_data(outlier_removed, method='minmax')
    print("Normalized DataFrame:")
    print(normalized_df)
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
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
    Calculate summary statistics for a DataFrame column.
    
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
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def main():
    # Example usage
    data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 14, 13, 12, 11]
    }
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df)
    print("\nSummary Statistics:")
    print(calculate_summary_stats(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned DataFrame (outliers removed):")
    print(cleaned_df)
    print("\nCleaned Summary Statistics:")
    print(calculate_summary_stats(cleaned_df, 'values'))

if __name__ == "__main__":
    main()