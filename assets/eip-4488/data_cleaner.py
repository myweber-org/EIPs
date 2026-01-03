
import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Columns to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): 'drop' to remove rows, 'fill' to fill values
        fill_value: Value to fill missing values with
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if df.empty:
        return df
    
    if strategy == 'drop':
        cleaned_df = df.dropna()
        removed_count = len(df) - len(cleaned_df)
        print(f"Removed {removed_count} rows with missing values")
    elif strategy == 'fill':
        cleaned_df = df.fillna(fill_value)
        filled_count = df.isna().sum().sum()
        print(f"Filled {filled_count} missing values")
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if validation passes
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def process_dataframe(df, operations=None):
    """
    Apply multiple cleaning operations to DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        operations (list): List of operations to apply
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if operations is None:
        operations = ['validate', 'remove_duplicates', 'clean_missing']
    
    cleaned_df = df.copy()
    
    for operation in operations:
        if operation == 'validate':
            validate_dataframe(cleaned_df)
        elif operation == 'remove_duplicates':
            cleaned_df = remove_duplicates(cleaned_df)
        elif operation == 'clean_missing':
            cleaned_df = clean_missing_values(cleaned_df, strategy='drop')
        else:
            print(f"Unknown operation: {operation}")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve', 'Frank'],
        'value': [10, 20, 20, 30, 40, 40, 50]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned = process_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned)