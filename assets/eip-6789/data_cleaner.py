import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to check for missing values.
                                 If None, checks all columns.
    
    Returns:
        pd.DataFrame: DataFrame with rows containing missing values removed.
    """
    if columns is None:
        columns = df.columns
    
    return df.dropna(subset=columns)

def fill_missing_with_mean(df, columns=None):
    """
    Fill missing values with column mean.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to fill. If None, fills all numeric columns.
    
    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filled = df.copy()
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df_filled[col] = df[col].fillna(df[col].mean())
    
    return df_filled

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to check for outliers.
                                 If None, checks all numeric columns.
        threshold (float): IQR multiplier for outlier detection.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to standardize.
                                 If None, standardizes all numeric columns.
    
    Returns:
        pd.DataFrame: DataFrame with standardized columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_standardized = df.copy()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df_standardized[col] = (df[col] - mean) / std
    
    return df_standardized

def clean_dataset(df, missing_strategy='remove', outlier_removal=True, standardization=True):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_strategy (str): Strategy for handling missing values.
                               'remove' or 'fill_mean'
        outlier_removal (bool): Whether to remove outliers
        standardization (bool): Whether to standardize numeric columns
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Handle missing values
    if missing_strategy == 'remove':
        df_clean = remove_missing_rows(df_clean)
    elif missing_strategy == 'fill_mean':
        df_clean = fill_missing_with_mean(df_clean)
    
    # Remove outliers
    if outlier_removal:
        df_clean = remove_outliers_iqr(df_clean)
    
    # Standardize columns
    if standardization:
        df_clean = standardize_columns(df_clean)
    
    return df_clean
import pandas as pd
import re

def clean_dataframe(df, text_column):
    """
    Clean a dataframe by removing duplicates and normalizing text in a specified column.
    """
    # Remove duplicate rows
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    # Normalize text: lowercase, remove extra whitespace
    def normalize_text(text):
        if pd.isna(text):
            return text
        text = str(text).lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    df_clean[text_column] = df_clean[text_column].apply(normalize_text)
    
    return df_clean

def save_cleaned_data(df, output_path):
    """
    Save the cleaned dataframe to a CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'text': ['Hello World', 'hello world', '  Python   Code  ', 'PYTHON CODE', 'Test', 'Test']
    }
    df = pd.DataFrame(sample_data)
    
    cleaned_df = clean_dataframe(df, 'text')
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    save_cleaned_data(cleaned_df, 'cleaned_data.csv')import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
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
    
    return filtered_df.reset_index(drop=True)

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
        'count': len(df[column]),
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

if __name__ == "__main__":
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 11, 10, 9, 8, 12, 13, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nOriginal Statistics:")
    print(calculate_basic_stats(df, 'values'))
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned Statistics:")
    print(calculate_basic_stats(cleaned_df, 'values'))
    
    normalized_df = normalize_column(cleaned_df, 'values', method='minmax')
    print("\nNormalized DataFrame:")
    print(normalized_df)