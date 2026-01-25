import pandas as pd

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing null values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
    
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

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        bool: True if validation passes, False otherwise.
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

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'name': ['Alice', 'Bob', 'Charlie', None, 'Alice'],
        'age': [25, 30, None, 35, 25],
        'score': [85.5, 92.0, 78.5, 88.0, 85.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataset(df)
    print(cleaned_df)
    
    # Validate the cleaned dataset
    is_valid = validate_dataset(cleaned_df, required_columns=['name', 'age', 'score'])
    print(f"\nDataset validation passed: {is_valid}")import pandas as pd

def clean_dataframe(df, drop_na=True, drop_duplicates=True):
    """
    Clean a pandas DataFrame by removing null values and duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_na (bool): If True, drop rows with any null values.
    drop_duplicates (bool): If True, drop duplicate rows.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def process_data_file(file_path, **kwargs):
    """
    Load and clean data from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    **kwargs: Additional arguments passed to clean_dataframe.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame or None if error occurs.
    """
    try:
        df = pd.read_csv(file_path)
        
        if not validate_dataframe(df):
            return None
        
        cleaned_df = clean_dataframe(df, **kwargs)
        return cleaned_df
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None