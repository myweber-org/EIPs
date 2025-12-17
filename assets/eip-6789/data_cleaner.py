import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_na_method='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): Columns to check for duplicates. 
            If None, checks all columns.
        fill_na_method (str): Method to fill missing values. 
            Options: 'mean', 'median', 'mode', or 'drop'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check is None:
        cleaned_df = cleaned_df.drop_duplicates()
    else:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    
    # Handle missing values
    numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
    
    if fill_na_method == 'mean':
        for col in numeric_cols:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
    elif fill_na_method == 'median':
        for col in numeric_cols:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_na_method == 'mode':
        for col in numeric_cols:
            mode_val = cleaned_df[col].mode()
            if not mode_val.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
    elif fill_na_method == 'drop':
        cleaned_df = cleaned_df.dropna(subset=numeric_cols)
    
    # For non-numeric columns, fill with most frequent value
    non_numeric_cols = cleaned_df.select_dtypes(exclude=['number']).columns
    for col in non_numeric_cols:
        mode_val = cleaned_df[col].mode()
        if not mode_val.empty:
            cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for basic data quality checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of columns that must be present.
    
    Returns:
        dict: Dictionary with validation results.
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
    
    return validation_results

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     sample_data = {
#         'id': [1, 2, 3, 4, 5, 5],
#         'value': [10.5, None, 15.2, 10.5, 8.7, 8.7],
#         'category': ['A', 'B', None, 'A', 'C', 'C']
#     }
#     
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     
#     cleaned = clean_dataset(df, columns_to_check=['id'], fill_na_method='mean')
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     
#     validation = validate_dataset(cleaned, required_columns=['id', 'value'])
#     print("\nValidation Results:")
#     for key, value in validation.items():
#         print(f"{key}: {value}")