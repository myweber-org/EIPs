import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_method='drop'):
    """
    Clean a pandas DataFrame by handling null values and optionally removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): If True, remove duplicate rows.
    fill_method (str): Method to handle nulls: 'drop' to remove rows, 'fill' to fill with column mean.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if fill_method == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_method == 'fill':
        for column in cleaned_df.select_dtypes(include=['number']).columns:
            cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mean())
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (bool, str) indicating success and message.
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame validation passed"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, None, 4, 4],
        'B': [5, None, 7, 8, 8],
        'C': [9, 10, 11, 12, 12]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataframe(df, drop_duplicates=True, fill_method='fill')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {is_valid} - {message}")