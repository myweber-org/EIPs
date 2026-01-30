
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    subset (list, optional): Columns to consider for duplicates.
    keep (str, optional): Which duplicates to keep.
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_column(df, column_name):
    """
    Clean a numeric column by removing non-numeric characters.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column_name (str): Name of column to clean.
    
    Returns:
    pd.DataFrame: DataFrame with cleaned column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list, optional): List of required columns.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if df.empty:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return Trueimport pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from a DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in specified columns using a given strategy.
    """
    df_filled = df.copy()
    if columns is None:
        columns = df.columns

    for col in columns:
        if df[col].dtype in [np.float64, np.int64]:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            else:
                fill_value = 0
            df_filled[col].fillna(fill_value, inplace=True)
        else:
            df_filled[col].fillna('Unknown', inplace=True)
    return df_filled

def normalize_column(df, column):
    """
    Normalize a numeric column to range [0, 1].
    """
    if df[column].dtype in [np.float64, np.int64]:
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val > min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataframe(df, duplicate_subset=None, fill_strategy='mean', normalize_columns=None):
    """
    Perform a complete cleaning pipeline on a DataFrame.
    """
    df_clean = remove_duplicates(df, subset=duplicate_subset)
    df_clean = fill_missing_values(df_clean, strategy=fill_strategy)
    
    if normalize_columns:
        for col in normalize_columns:
            df_clean = normalize_column(df_clean, col)
    
    return df_clean

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5],
        'B': [5.0, np.nan, 7.0, 8.0, 9.0],
        'C': ['x', 'y', 'y', 'z', np.nan]
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataframe(df, duplicate_subset=['A'], fill_strategy='mean', normalize_columns=['B'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)