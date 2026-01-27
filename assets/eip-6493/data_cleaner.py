
import pandas as pd
import numpy as np

def remove_missing_rows(df, threshold=0.5):
    """
    Remove rows with missing values exceeding threshold percentage.
    
    Args:
        df: pandas DataFrame
        threshold: float between 0 and 1
    
    Returns:
        Cleaned DataFrame
    """
    missing_per_row = df.isnull().mean(axis=1)
    return df[missing_per_row <= threshold].reset_index(drop=True)

def fill_missing_with_median(df, columns=None):
    """
    Fill missing values with column median.
    
    Args:
        df: pandas DataFrame
        columns: list of column names or None for all numeric columns
    
    Returns:
        DataFrame with filled values
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filled = df.copy()
    for col in columns:
        if col in df.columns:
            median_val = df[col].median()
            df_filled[col] = df[col].fillna(median_val)
    
    return df_filled

def detect_outliers_iqr(df, column):
    """
    Detect outliers using IQR method.
    
    Args:
        df: pandas DataFrame
        column: column name to check
    
    Returns:
        Series of boolean values indicating outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def cap_outliers(df, column, method='iqr'):
    """
    Cap outliers to specified bounds.
    
    Args:
        df: pandas DataFrame
        column: column name to process
        method: 'iqr' or 'percentile'
    
    Returns:
        DataFrame with capped values
    """
    df_capped = df.copy()
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    elif method == 'percentile':
        lower_bound = df[column].quantile(0.01)
        upper_bound = df[column].quantile(0.99)
    else:
        raise ValueError("Method must be 'iqr' or 'percentile'")
    
    df_capped[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df_capped

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to zero mean and unit variance.
    
    Args:
        df: pandas DataFrame
        columns: list of column names or None for all numeric columns
    
    Returns:
        Standardized DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_standardized = df.copy()
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df_standardized[col] = (df[col] - mean_val) / std_val
    
    return df_standardizedimport pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Columns to consider for duplicates.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        strategy (str): 'mean', 'median', 'mode', or 'constant'.
        columns (list): Columns to fill. If None, fill all numeric columns.
    
    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if strategy == 'mean':
            df_filled[col].fillna(df[col].mean(), inplace=True)
        elif strategy == 'median':
            df_filled[col].fillna(df[col].median(), inplace=True)
        elif strategy == 'mode':
            df_filled[col].fillna(df[col].mode()[0], inplace=True)
        elif strategy == 'constant':
            df_filled[col].fillna(0, inplace=True)
    
    return df_filled

def normalize_columns(df, columns=None, method='minmax'):
    """
    Normalize specified columns in DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): Columns to normalize.
        method (str): 'minmax' or 'zscore'.
    
    Returns:
        pd.DataFrame: DataFrame with normalized columns.
    """
    df_normalized = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df_normalized[col] = (df[col] - mean_val) / std_val
    
    return df_normalized

def clean_dataset(df, remove_dups=True, fill_na=True, normalize=True):
    """
    Apply a complete cleaning pipeline to the dataset.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        remove_dups (bool): Whether to remove duplicates.
        fill_na (bool): Whether to fill missing values.
        normalize (bool): Whether to normalize numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if fill_na:
        cleaned_df = fill_missing_values(cleaned_df, strategy='mean')
    
    if normalize:
        cleaned_df = normalize_columns(cleaned_df, method='minmax')
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, None],
        'B': [10, 20, 20, 40, 50, 60],
        'C': [100, 200, 300, 400, 500, 600]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataset(df)
    print(cleaned)