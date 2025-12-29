import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_na=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.

    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_na (str, dict, or None): Method to fill missing values. 
                                   Options: 'ffill', 'bfill', a scalar value, 
                                   or a dict of column:value pairs. Default is None.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if fill_na is not None:
        cleaned_df.fillna(fill_na, inplace=True)
    
    if drop_duplicates:
        cleaned_df.drop_duplicates(inplace=True)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate the DataFrame for required columns and basic integrity.

    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.

    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    if df.empty:
        print("DataFrame is empty.")
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, None, 4, 2],
        'B': [5, None, 7, 8, 5],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_na=0)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nData validation passed: {is_valid}")