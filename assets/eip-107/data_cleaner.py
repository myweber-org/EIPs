
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, standardize_columns=True):
    """
    Clean a pandas DataFrame by removing duplicates and standardizing column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to remove duplicate rows.
        standardize_columns (bool): Whether to standardize column names.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if standardize_columns:
        cleaned_df.columns = cleaned_df.columns.str.strip().str.lower().str.replace(' ', '_')
        print("Column names standardized.")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        dict: Validation results.
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'null_counts': {},
        'dtypes': {}
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['is_valid'] = False
            validation_results['missing_columns'] = missing
    
    for column in df.columns:
        null_count = df[column].isnull().sum()
        validation_results['null_counts'][column] = null_count
        validation_results['dtypes'][column] = str(df[column].dtype)
    
    return validation_results

def sample_data():
    """
    Create sample data for testing.
    
    Returns:
        pd.DataFrame: Sample DataFrame with test data.
    """
    data = {
        'User ID': [1, 2, 3, 1, 2, 4],
        'First Name': ['John', 'Jane', 'Bob', 'John', 'Jane', 'Alice'],
        'Last Name': ['Doe', 'Smith', 'Johnson', 'Doe', 'Smith', 'Brown'],
        'Email': ['john@example.com', 'jane@example.com', 'bob@example.com', 
                  'john@example.com', 'jane@example.com', 'alice@example.com'],
        'Age': [25, 30, 35, 25, 30, 28],
        'Salary': [50000, 60000, 70000, 50000, 60000, 55000]
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Example usage
    df = sample_data()
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    validation = validate_dataframe(cleaned_df, required_columns=['user_id', 'email'])
    print("Validation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")