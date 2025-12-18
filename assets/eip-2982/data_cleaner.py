import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, convert_types=True):
    """
    Clean a pandas DataFrame by removing duplicates, filling missing values,
    and converting data types where appropriate.
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        initial_rows = df_clean.shape[0]
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - df_clean.shape[0]
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        object_cols = df_clean.select_dtypes(include=['object']).columns
        for col in object_cols:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna('Unknown')
    
    if convert_types:
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col])
                    print(f"Converted column '{col}' to datetime.")
                except (ValueError, TypeError):
                    pass
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    
    if df.empty:
        raise ValueError("DataFrame is empty.")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, None, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', None, 'A'],
        'date': ['2023-01-01', '2023-01-02', '2023-01-02', 'invalid', '2023-01-04', '2023-01-05']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaning DataFrame...")
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    try:
        validate_dataframe(cleaned_df, required_columns=['id', 'value'])
        print("\nDataFrame validation passed.")
    except Exception as e:
        print(f"\nValidation error: {e}")