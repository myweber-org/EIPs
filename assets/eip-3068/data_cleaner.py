
import pandas as pd

def clean_dataframe(df, fill_strategy='drop', column_case='lower'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    fill_strategy (str): Strategy for handling null values. 
                         Options: 'drop' to remove rows, 'fill' to fill with column mean (numeric) or mode (categorical).
    column_case (str): Target case for column names. Options: 'lower', 'upper', 'title'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # Standardize column names
    if column_case == 'lower':
        df_clean.columns = df_clean.columns.str.lower()
    elif column_case == 'upper':
        df_clean.columns = df_clean.columns.str.upper()
    elif column_case == 'title':
        df_clean.columns = df_clean.columns.str.title()
    
    # Handle missing values
    if fill_strategy == 'drop':
        df_clean = df_clean.dropna()
    elif fill_strategy == 'fill':
        for col in df_clean.columns:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else '', inplace=True)
    
    # Reset index after cleaning
    df_clean.reset_index(drop=True, inplace=True)
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if not isinstance(df, pd.DataFrame):
        validation_result['is_valid'] = False
        validation_result['errors'].append('Input is not a pandas DataFrame')
        return validation_result
    
    if df.empty:
        validation_result['warnings'].append('DataFrame is empty')
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f'Missing required columns: {missing_columns}')
    
    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        validation_result['warnings'].append('Duplicate column names detected')
    
    return validation_result

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'Name': ['Alice', 'Bob', None, 'David'],
        'Age': [25, None, 30, 35],
        'Score': [85.5, 92.0, 78.5, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned_df = clean_dataframe(df, fill_strategy='fill', column_case='lower')
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    # Validate the cleaned data
    validation = validate_dataframe(cleaned_df, required_columns=['name', 'age'])
    print("Validation Result:")
    print(validation)