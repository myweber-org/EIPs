import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str): Strategy to fill missing values. 
                           Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median', 'mode']:
        for column in cleaned_df.select_dtypes(include=['float64', 'int64']).columns:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean':
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                elif fill_missing == 'median':
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                elif fill_missing == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        tuple: (is_valid, message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': [10, 11, 12, 12, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='median')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {message}")
import pandas as pd
import numpy as np

def clean_dataframe(df, column_names=None, remove_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    column_names (list): Specific columns to clean, if None clean all object columns
    remove_duplicates (bool): Whether to remove duplicate rows
    normalize_text (bool): Whether to normalize text columns (lowercase, strip whitespace)
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    cleaned_df = df.copy()
    
    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if normalize_text:
        if column_names is None:
            text_columns = cleaned_df.select_dtypes(include=['object']).columns
        else:
            text_columns = [col for col in column_names if col in cleaned_df.columns]
        
        for col in text_columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.lower().str.strip()
            cleaned_df[col] = cleaned_df[col].replace('nan', np.nan)
            cleaned_df[col] = cleaned_df[col].replace('none', np.nan)
        
        print(f"Normalized {len(text_columns)} text columns")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, check_missing=True):
    """
    Validate DataFrame structure and data quality.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of columns that must be present
    check_missing (bool): Whether to check for missing values
    
    Returns:
    dict: Dictionary with validation results
    """
    
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': {},
        'column_types': {},
        'validation_passed': True
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['missing_columns'] = missing_cols
            validation_results['validation_passed'] = False
    
    if check_missing:
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                validation_results['missing_values'][col] = missing_count
    
    for col in df.columns:
        validation_results['column_types'][col] = str(df[col].dtype)
    
    return validation_results

def sample_data_for_testing():
    """Create sample DataFrame for testing the cleaning functions."""
    
    data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson', 'Alice Brown', ''],
        'email': ['john@example.com', 'jane@example.com', 'JOHN@example.com', 'bob@example.com', None, 'test@example.com'],
        'age': [25, 30, 25, 35, 28, 40],
        'city': ['New York', 'Los Angeles', 'new york', 'Chicago', 'Boston', 'Chicago']
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Example usage
    df = sample_data_for_testing()
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataframe(df)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    validation = validate_dataframe(cleaned_df, required_columns=['name', 'email', 'age'])
    print("Validation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")