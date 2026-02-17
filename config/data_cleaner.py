import pandas as pd
import numpy as np

def clean_csv_data(filepath, missing_strategy='mean', columns_to_drop=None):
    """
    Load and clean CSV data by handling missing values and optionally dropping columns.
    
    Parameters:
    filepath (str): Path to the CSV file.
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop').
    columns_to_drop (list): List of column names to drop from the dataset.
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {filepath} was not found.")
    except pd.errors.EmptyDataError:
        raise ValueError("The provided CSV file is empty.")
    
    original_shape = df.shape
    
    if columns_to_drop:
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    if missing_strategy == 'drop':
        df = df.dropna()
    elif missing_strategy in ['mean', 'median']:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if missing_strategy == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        else:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif missing_strategy == 'mode':
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    print(f"Data cleaning completed. Original shape: {original_shape}, Cleaned shape: {df.shape}")
    print(f"Removed {original_shape[0] - df.shape[0]} rows and {original_shape[1] - df.shape[1]} columns.")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate the DataFrame for required columns and basic integrity.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    bool: True if validation passes, raises ValueError otherwise.
    """
    if df.empty:
        raise ValueError("DataFrame is empty after cleaning.")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.isnull().sum().sum() > 0:
        print("Warning: DataFrame still contains missing values.")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [5, np.nan, np.nan, 8, 10],
        'C': ['x', 'y', 'z', np.nan, 'x'],
        'D': [10, 20, 30, 40, 50]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', missing_strategy='mean', columns_to_drop=['D'])
    
    try:
        validate_dataframe(cleaned_df, required_columns=['A', 'B'])
        print("Data validation passed.")
    except ValueError as e:
        print(f"Validation error: {e}")
    
    import os
    os.remove('test_data.csv')