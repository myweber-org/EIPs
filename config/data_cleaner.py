import pandas as pd

def clean_dataset(df, drop_duplicates=True, fillna_method=None, fillna_value=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        fillna_method (str): Method to fill missing values ('ffill', 'bfill', etc.).
        fillna_value: Specific value to fill missing values with.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fillna_method is not None:
        cleaned_df = cleaned_df.fillna(method=fillna_method)
    elif fillna_value is not None:
        cleaned_df = cleaned_df.fillna(fillna_value)
    
    # Remove duplicates
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        dict: Validation results with status and messages.
    """
    validation_result = {
        'is_valid': True,
        'messages': [],
        'missing_columns': []
    }
    
    # Check for required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_result['is_valid'] = False
            validation_result['missing_columns'] = missing_cols
            validation_result['messages'].append(f"Missing required columns: {missing_cols}")
    
    # Check for empty DataFrame
    if df.empty:
        validation_result['is_valid'] = False
        validation_result['messages'].append("DataFrame is empty")
    
    # Check data types
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        validation_result['messages'].append("No numeric columns found in dataset")
    
    return validation_result

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {
#         'A': [1, 2, None, 4, 2],
#         'B': [5, 6, 7, None, 6],
#         'C': ['x', 'y', 'z', 'x', 'y']
#     }
#     df = pd.DataFrame(sample_data)
#     cleaned = clean_dataset(df, fillna_value=0)
#     validation = validate_dataset(cleaned, required_columns=['A', 'B'])
#     print(f"Cleaned shape: {cleaned.shape}")
#     print(f"Validation result: {validation}")