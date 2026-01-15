import pandas as pd

def clean_dataset(df):
    """
    Remove null values and duplicate rows from a pandas DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame to be cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Remove duplicate rows
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def filter_by_column(df, column_name, min_value=None, max_value=None):
    """
    Filter DataFrame based on column value range.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Column to filter by.
        min_value: Minimum value for filtering (inclusive).
        max_value: Maximum value for filtering (inclusive).
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    filtered_df = df.copy()
    
    if min_value is not None:
        filtered_df = filtered_df[filtered_df[column_name] >= min_value]
    
    if max_value is not None:
        filtered_df = filtered_df[filtered_df[column_name] <= max_value]
    
    return filtered_df