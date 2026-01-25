import pandas as pd

def clean_dataframe(df, drop_na=True, column_case='lower'):
    """
    Clean a pandas DataFrame by handling null values and standardizing column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_na (bool): If True, drop rows with any null values. Default True.
        column_case (str): Target case for column names ('lower', 'upper', or 'title'). Default 'lower'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle null values
    if drop_na:
        cleaned_df = cleaned_df.dropna()
    else:
        # Fill numeric columns with median, categorical with mode
        for col in cleaned_df.columns:
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
    
    # Standardize column names
    if column_case == 'lower':
        cleaned_df.columns = cleaned_df.columns.str.lower()
    elif column_case == 'upper':
        cleaned_df.columns = cleaned_df.columns.str.upper()
    elif column_case == 'title':
        cleaned_df.columns = cleaned_df.columns.str.title()
    
    # Remove extra whitespace from string columns
    for col in cleaned_df.select_dtypes(include=['object']).columns:
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {
#         'Name': ['Alice', 'Bob', None, 'Charlie'],
#         'Age': [25, None, 30, 35],
#         'City': ['NYC', 'LA', 'Chicago', None]
#     }
#     df = pd.DataFrame(sample_data)
#     cleaned = clean_dataframe(df, drop_na=False)
#     print(cleaned)