import pandas as pd
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

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): 'mean', 'median', 'mode', or 'constant'
        columns (list): Specific columns to fill
    
    Returns:
        pd.DataFrame: DataFrame with filled values
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else np.nan
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df_filled[col] = df[col].fillna(fill_value)
        else:
            df_filled[col] = df[col].fillna('Unknown')
    
    return df_filled

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to normalize
        method (str): 'minmax' or 'zscore'
    
    Returns:
        pd.DataFrame: DataFrame with normalized column
    """
    df_normalized = df.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df_normalized[column] = (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val != 0:
            df_normalized[column] = (df[column] - mean_val) / std_val
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return df_normalized

def clean_dataframe(df, operations=None):
    """
    Apply multiple cleaning operations to DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        operations (list): List of operation dictionaries
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if operations is None:
        operations = [
            {'function': remove_duplicates, 'args': {}},
            {'function': fill_missing_values, 'args': {'strategy': 'mean'}}
        ]
    
    for operation in operations:
        func = operation['function']
        args = operation.get('args', {})
        cleaned_df = func(cleaned_df, **args)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, np.nan],
        'B': [10, 20, 20, 40, 50, 60],
        'C': ['a', 'b', 'b', 'd', 'e', 'f']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataframe(df)
    print(cleaned)