import pandas as pd

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing null values and duplicate rows.
    
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

def filter_by_column(df, column_name, threshold):
    """
    Filter DataFrame rows based on column value threshold.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of column to filter by.
        threshold (float): Threshold value for filtering.
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    filtered_df = df[df[column_name] >= threshold]
    return filtered_df.reset_index(drop=True)