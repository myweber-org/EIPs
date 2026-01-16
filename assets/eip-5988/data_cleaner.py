import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, handle_nulls='drop', fill_value=None):
    """
    Clean a pandas DataFrame by handling duplicates and null values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        handle_nulls (str): Method to handle nulls - 'drop', 'fill', or 'ignore'.
        fill_value: Value to fill nulls with if handle_nulls is 'fill'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if handle_nulls == 'drop':
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.dropna()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} rows with null values.")
    elif handle_nulls == 'fill' and fill_value is not None:
        null_count = cleaned_df.isnull().sum().sum()
        cleaned_df = cleaned_df.fillna(fill_value)
        print(f"Filled {null_count} null values with {fill_value}.")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame."
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows."
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid."

def sample_data_cleaning():
    """Example usage of the data cleaning functions."""
    data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', None],
        'score': [85, 92, 92, 78, None, 88]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(
        df, 
        drop_duplicates=True, 
        handle_nulls='fill', 
        fill_value=0
    )
    
    print("Cleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(
        cleaned, 
        required_columns=['id', 'name', 'score'], 
        min_rows=1
    )
    
    print(f"\nValidation: {message}")
    return cleaned

if __name__ == "__main__":
    sample_data_cleaning()