
import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, remove_duplicates=True):
    """
    Clean a pandas DataFrame by standardizing column names and removing duplicates.
    
    Args:
        df: pandas DataFrame to clean
        column_mapping: Optional dictionary mapping original column names to standardized names
        remove_duplicates: Boolean indicating whether to remove duplicate rows
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    # Standardize column names
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    # Convert column names to lowercase and replace spaces with underscores
    cleaned_df.columns = cleaned_df.columns.str.lower().str.replace(' ', '_')
    
    # Remove duplicate rows if specified
    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed_count = initial_rows - len(cleaned_df)
        print(f"Removed {removed_count} duplicate rows")
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        Boolean indicating whether validation passed
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def sample_data_cleaning():
    """Example usage of the data cleaning functions."""
    # Create sample data
    data = {
        'Customer ID': [1, 2, 3, 1, 4],
        'First Name': ['John', 'Jane', 'Bob', 'John', 'Alice'],
        'Last Name': ['Doe', 'Smith', 'Johnson', 'Doe', 'Williams'],
        'Email': ['john@example.com', 'jane@example.com', 'bob@example.com', 'john@example.com', 'alice@example.com']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    column_mapping = {'Customer ID': 'customer_id', 'First Name': 'first_name', 'Last Name': 'last_name'}
    cleaned_df = clean_dataset(df, column_mapping=column_mapping)
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    is_valid = validate_dataframe(cleaned_df, required_columns=['customer_id', 'email'])
    print(f"\nData validation passed: {is_valid}")
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = sample_data_cleaning()