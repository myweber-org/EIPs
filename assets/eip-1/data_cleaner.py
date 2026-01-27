import pandas as pd

def clean_dataset(df, drop_duplicates=True):
    """
    Clean a pandas DataFrame by removing null values and optionally duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.dropna()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns):
    """
    Validate that the DataFrame contains all required columns.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if all required columns are present, False otherwise.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'name': ['Alice', 'Bob', None, 'Alice', 'Charlie'],
        'age': [25, 30, 35, 25, None],
        'city': ['NYC', 'LA', 'NYC', 'NYC', 'Chicago']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataset(df)
    print(cleaned)
    
    required_cols = ['name', 'age']
    is_valid = validate_data(cleaned, required_cols)
    print(f"\nData validation result: {is_valid}")