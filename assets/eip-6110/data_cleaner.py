
import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_na_method='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    columns_to_check (list, optional): List of columns to check for duplicates.
                                       If None, checks all columns.
    fill_na_method (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
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
            cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
    elif fill_na_method == 'median':
        for col in numeric_cols:
            cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_na_method == 'mode':
        for col in numeric_cols:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0, inplace=True)
    elif fill_na_method == 'drop':
        cleaned_df = cleaned_df.dropna(subset=numeric_cols)
    
    # For non-numeric columns, fill with most frequent value
    non_numeric_cols = cleaned_df.select_dtypes(exclude=['number']).columns
    for col in non_numeric_cols:
        if not cleaned_df[col].mode().empty:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
        else:
            cleaned_df[col].fillna('Unknown', inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list, optional): List of columns that must be present.
    
    Returns:
    tuple: (is_valid, message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'id': [1, 2, 2, 3, 4, 5],
#         'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve'],
#         'age': [25, 30, 30, None, 35, 40],
#         'score': [85.5, 92.0, 92.0, 78.5, None, 95.0]
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     print("\nCleaned DataFrame (using mean for missing values):")
#     cleaned = clean_dataset(df, fill_na_method='mean')
#     print(cleaned)
#     
#     # Validate
#     is_valid, message = validate_dataframe(cleaned, ['id', 'name', 'age', 'score'])
#     print(f"\nValidation: {message}")