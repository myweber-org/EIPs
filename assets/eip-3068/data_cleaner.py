import pandas as pd
import numpy as np
from typing import List, Optional

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase with underscores."""
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def remove_duplicate_rows(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """Remove duplicate rows based on specified columns or all columns."""
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Fill missing values using specified strategy."""
    df_filled = df.copy()
    
    if columns is None:
        columns = df.columns.tolist()
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
            elif strategy == 'median':
                df_filled[col] = df_filled[col].fillna(df_filled[col].median())
            elif strategy == 'zero':
                df_filled[col] = df_filled[col].fillna(0)
        else:
            df_filled[col] = df_filled[col].fillna('unknown')
    
    return df_filled

def standardize_text_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Standardize text columns to lowercase and strip whitespace."""
    df_std = df.copy()
    for col in columns:
        if col in df_std.columns:
            df_std[col] = df_std[col].astype(str).str.lower().str.strip()
    return df_std

def clean_dataset(df: pd.DataFrame, text_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Apply all cleaning functions to dataset."""
    df_clean = df.copy()
    df_clean = clean_column_names(df_clean)
    df_clean = remove_duplicate_rows(df_clean)
    
    if text_columns:
        df_clean = standardize_text_columns(df_clean, text_columns)
    
    df_clean = fill_missing_values(df_clean)
    return df_clean

if __name__ == "__main__":
    sample_data = {
        'Product Name': ['Laptop', 'Mouse', 'Laptop', 'Keyboard', None],
        'Price': [999.99, 25.50, 999.99, 45.75, 30.00],
        'Category': ['Electronics', 'Electronics', 'electronics', 'Electronics', 'Accessories']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataset(df, text_columns=['Product Name', 'Category'])
    print(cleaned_df)
import pandas as pd
import numpy as np

def clean_dataset(df, text_columns=None, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing text columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    text_columns (list): List of column names containing text data.
    fill_strategy (str): Strategy for filling numeric missing values ('mean', 'median', 'mode').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values in numeric columns
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if cleaned_df[col].isnull().any():
            if fill_strategy == 'mean':
                fill_value = cleaned_df[col].mean()
            elif fill_strategy == 'median':
                fill_value = cleaned_df[col].median()
            elif fill_strategy == 'mode':
                fill_value = cleaned_df[col].mode()[0]
            else:
                fill_value = 0
            cleaned_df[col].fillna(fill_value, inplace=True)
    
    # Standardize text columns
    if text_columns:
        for col in text_columns:
            if col in cleaned_df.columns:
                # Convert to string, strip whitespace, and convert to lowercase
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
                # Replace empty strings with NaN then fill with 'unknown'
                cleaned_df[col].replace(['', 'nan', 'none'], np.nan, inplace=True)
                cleaned_df[col].fillna('unknown', inplace=True)
    
    # Reset index after cleaning
    cleaned_df.reset_index(drop=True, inplace=True)
    
    return cleaned_df

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    subset (list): Columns to consider for identifying duplicates.
    keep (str): Which duplicates to keep ('first', 'last', False).
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of columns that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['John', 'Jane', None, 'Bob', 'Alice'],
        'age': [25, 30, np.nan, 35, 40],
        'score': [85.5, 92.0, 78.5, np.nan, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned = clean_dataset(df, text_columns=['name'], fill_strategy='mean')
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n")
    
    # Validate the cleaned data
    is_valid, message = validate_data(cleaned, required_columns=['id', 'name', 'age'])
    print(f"Validation: {is_valid} - {message}")