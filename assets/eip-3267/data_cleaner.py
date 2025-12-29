import pandas as pd

def clean_dataset(df, columns_to_check=None):
    """
    Clean a pandas DataFrame by removing null values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): Specific columns to check for nulls.
            If None, checks all columns. Defaults to None.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove rows with null values
    if columns_to_check:
        cleaned_df = cleaned_df.dropna(subset=columns_to_check)
    else:
        cleaned_df = cleaned_df.dropna()
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns):
    """
    Validate that required columns exist in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        bool: True if all required columns exist, False otherwise.
    """
    existing_columns = set(df.columns)
    required_set = set(required_columns)
    
    return required_set.issubset(existing_columns)

# Example usage (commented out)
# if __name__ == "__main__":
#     sample_data = {
#         'id': [1, 2, 3, 4, 5, 5],
#         'name': ['Alice', 'Bob', None, 'David', 'Eve', 'Eve'],
#         'age': [25, 30, 35, None, 40, 40]
#     }
#     
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     
#     cleaned = clean_dataset(df)
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     
#     is_valid = validate_data(cleaned, ['id', 'name'])
#     print(f"\nData validation result: {is_valid}")
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, strategy='mean'):
    """
    Clean a CSV file by handling missing values.
    
    Parameters:
    input_path (str): Path to the input CSV file
    output_path (str, optional): Path to save cleaned CSV. If None, returns DataFrame
    strategy (str): Method for handling missing values: 'mean', 'median', 'mode', or 'drop'
    
    Returns:
    pd.DataFrame or None: Cleaned DataFrame if output_path is None, else None
    """
    
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    
    if strategy == 'drop':
        df_cleaned = df.dropna()
    elif strategy == 'mean':
        df_cleaned = df.fillna(df.select_dtypes(include=[np.number]).mean())
    elif strategy == 'median':
        df_cleaned = df.fillna(df.select_dtypes(include=[np.number]).median())
    elif strategy == 'mode':
        df_cleaned = df.fillna(df.mode().iloc[0])
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'drop', 'mean', 'median', or 'mode'")
    
    new_shape = df_cleaned.shape
    rows_removed = original_shape[0] - new_shape[0] if strategy == 'drop' else 0
    
    print(f"Cleaned data shape: {new_shape}")
    print(f"Rows removed: {rows_removed}")
    print(f"Missing values after cleaning: {df_cleaned.isnull().sum().sum()}")
    
    if output_path:
        df_cleaned.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        return None
    else:
        return df_cleaned

def validate_csv_columns(input_path, required_columns):
    """
    Validate that a CSV file contains required columns.
    
    Parameters:
    input_path (str): Path to the CSV file
    required_columns (list): List of required column names
    
    Returns:
    tuple: (bool, list) - (is_valid, missing_columns)
    """
    
    df = pd.read_csv(input_path)
    existing_columns = set(df.columns)
    required_set = set(required_columns)
    
    missing_columns = list(required_set - existing_columns)
    is_valid = len(missing_columns) == 0
    
    return is_valid, missing_columns

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'temperature': [22.5, np.nan, 24.0, np.nan, 23.0],
        'humidity': [45.0, 50.0, np.nan, 48.0, 47.0],
        'pressure': [1013.2, 1012.8, 1013.5, np.nan, 1013.0]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    print("Testing data cleaning utility...")
    cleaned_df = clean_csv_data('test_data.csv', strategy='mean')
    
    is_valid, missing = validate_csv_columns('test_data.csv', ['id', 'temperature', 'humidity'])
    print(f"CSV validation: {is_valid}, missing columns: {missing}")
    
    import os
    if os.path.exists('test_data.csv'):
        os.remove('test_data.csv')