
import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: Boolean indicating whether to drop duplicate rows
        fill_missing: Boolean indicating whether to fill missing values
        fill_value: Value to use for filling missing data
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        Boolean indicating whether validation passed
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if df.empty:
        print("DataFrame is empty")
        return False
    
    return True

def remove_outliers(df, column, threshold=3):
    """
    Remove outliers from a DataFrame column using z-score method.
    
    Args:
        df: pandas DataFrame
        column: Column name to check for outliers
        threshold: Z-score threshold for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    from scipy import stats
    import numpy as np
    
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    filtered_entries = z_scores < threshold
    return df[filtered_entries]