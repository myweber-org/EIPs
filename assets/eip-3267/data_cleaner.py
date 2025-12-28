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