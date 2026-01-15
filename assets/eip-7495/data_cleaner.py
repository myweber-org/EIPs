
import pandas as pd
import numpy as np

def clean_dataset(df, text_columns=None, fill_na=True):
    """
    Clean a pandas DataFrame by handling missing values and standardizing text.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        text_columns (list): List of column names containing text data
        fill_na (bool): Whether to fill missing values
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if fill_na:
        for col in df_clean.columns:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif df_clean[col].dtype == 'object':
                df_clean[col].fillna('unknown', inplace=True)
    
    if text_columns:
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()
    
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    if fill_na:
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
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
    
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        validation_result['warnings'].append(f'Found {duplicate_rows} duplicate rows')
    
    null_counts = df.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0].index.tolist()
    if columns_with_nulls:
        validation_result['warnings'].append(f'Columns with null values: {columns_with_nulls}')
    
    return validation_result

if __name__ == "__main__":
    sample_data = {
        'name': ['Alice', 'Bob', None, 'David', 'Eve'],
        'age': [25, 30, np.nan, 35, 40],
        'score': [85.5, 92.0, 78.5, np.inf, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50)
    
    cleaned_df = clean_dataset(df, text_columns=['name'])
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    validation = validate_dataframe(cleaned_df, required_columns=['name', 'age'])
    print("\nValidation Results:")
    print(validation)