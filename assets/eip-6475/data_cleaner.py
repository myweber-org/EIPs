
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from specified column using IQR method.
    
    Args:
        df: pandas DataFrame
        column: column name to process
    
    Returns:
        Cleaned DataFrame with outliers removed
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

def normalize_column(df, column, method='minmax'):
    """
    Normalize specified column using selected method.
    
    Args:
        df: pandas DataFrame
        column: column name to normalize
        method: normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized column
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
        if std_val > 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_copy

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Args:
        df: pandas DataFrame
        strategy: imputation strategy ('mean', 'median', or 'drop')
    
    Returns:
        DataFrame with handled missing values
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        df_copy = df_copy.dropna(subset=numeric_cols)
    
    elif strategy == 'mean':
        for col in numeric_cols:
            df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    
    elif strategy == 'median':
        for col in numeric_cols:
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    
    else:
        raise ValueError("Strategy must be 'mean', 'median', or 'drop'")
    
    return df_copy

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', 
                  normalize_method='minmax', missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        numeric_columns: list of numeric columns to process
        outlier_method: outlier removal method
        normalize_method: normalization method
        missing_strategy: missing value handling strategy
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    cleaned_df = handle_missing_values(cleaned_df, strategy=missing_strategy)
    
    if outlier_method == 'iqr':
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = normalize_column(cleaned_df, col, method=normalize_method)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],
        'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'C': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataset(df)
    print(cleaned)
import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_method='iqr', columns=None):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for missing values ('mean', 'median', 'mode', 'drop').
    outlier_method (str): Method for outlier detection ('iqr', 'zscore').
    columns (list): Specific columns to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if missing_strategy != 'drop':
            if missing_strategy == 'mean':
                fill_value = df_clean[col].mean()
            elif missing_strategy == 'median':
                fill_value = df_clean[col].median()
            elif missing_strategy == 'mode':
                fill_value = df_clean[col].mode()[0]
            else:
                fill_value = 0
            df_clean[col].fillna(fill_value, inplace=True)
        else:
            df_clean = df_clean.dropna(subset=[col])
        
        if outlier_method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = np.where((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound), 
                                     df_clean[col].median(), df_clean[col])
        elif outlier_method == 'zscore':
            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()
            z_scores = np.abs((df_clean[col] - mean_val) / std_val)
            df_clean[col] = np.where(z_scores > 3, mean_val, df_clean[col])
    
    return df_clean

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    subset (list): Columns to consider for duplicates.
    keep (str): Which duplicates to keep ('first', 'last', False).
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def normalize_columns(df, columns=None, method='minmax'):
    """
    Normalize specified columns in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): Columns to normalize.
    method (str): Normalization method ('minmax', 'standard').
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns.
    """
    df_norm = df.copy()
    
    if columns is None:
        numeric_cols = df_norm.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    for col in columns:
        if col not in df_norm.columns:
            continue
            
        if method == 'minmax':
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val != min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        elif method == 'standard':
            mean_val = df_norm[col].mean()
            std_val = df_norm[col].std()
            if std_val != 0:
                df_norm[col] = (df_norm[col] - mean_val) / std_val
    
    return df_norm

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': [10, 11, 12, 13, 14]
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataframe(df, missing_strategy='median', outlier_method='iqr')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
import pandas as pd

def clean_dataset(df, column_name):
    """
    Clean a specific column in a DataFrame by removing duplicates,
    stripping whitespace, and converting to lowercase.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    # Remove duplicates
    df = df.drop_duplicates(subset=[column_name])

    # Normalize strings: strip whitespace and convert to lowercase
    df[column_name] = df[column_name].astype(str).str.strip().str.lower()

    return df

def main():
    # Example usage
    data = {'Name': [' Alice ', 'bob', 'Alice', 'Bob ', 'Charlie']}
    df = pd.DataFrame(data)

    print("Original DataFrame:")
    print(df)

    cleaned_df = clean_dataset(df, 'Name')
    print("\nCleaned DataFrame:")
    print(cleaned_df)

if __name__ == "__main__":
    main()