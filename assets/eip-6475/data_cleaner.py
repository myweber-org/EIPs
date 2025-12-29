import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[col].median()
                elif fill_missing == 'zero':
                    fill_value = 0
                else:
                    fill_value = fill_missing
                
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled missing values in column '{col}' with {fill_value}.")
    
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if cleaned_df[col].isnull().any():
            cleaned_df[col] = cleaned_df[col].fillna('Unknown')
            print(f"Filled missing values in column '{col}' with 'Unknown'.")
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate the dataset for required columns and basic integrity.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("DataFrame is empty.")
    
    print(f"Data validation passed. Shape: {df.shape}")
    return True

def main():
    # Example usage
    data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', None, 'A'],
        'score': [85, 92, 92, 78, 88, np.nan]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    try:
        validate_data(cleaned_df, required_columns=['id', 'value', 'category'])
    except ValueError as e:
        print(f"Validation error: {e}")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    
    df_cleaned = df.copy()
    
    if drop_duplicates:
        initial_rows = df_cleaned.shape[0]
        df_cleaned = df_cleaned.drop_duplicates()
        removed = initial_rows - df_cleaned.shape[0]
        print(f"Removed {removed} duplicate rows.")
    
    missing_before = df_cleaned.isnull().sum().sum()
    
    if fill_missing == 'drop':
        df_cleaned = df_cleaned.dropna()
        print(f"Dropped rows with missing values. {missing_before} missing values removed.")
    elif fill_missing in ['mean', 'median', 'mode']:
        for column in df_cleaned.select_dtypes(include=[np.number]).columns:
            if df_cleaned[column].isnull().any():
                if fill_missing == 'mean':
                    fill_value = df_cleaned[column].mean()
                elif fill_missing == 'median':
                    fill_value = df_cleaned[column].median()
                elif fill_missing == 'mode':
                    fill_value = df_cleaned[column].mode()[0]
                df_cleaned[column].fillna(fill_value, inplace=True)
        print(f"Filled missing values using {fill_missing} method.")
    
    missing_after = df_cleaned.isnull().sum().sum()
    print(f"Missing values before: {missing_before}, after: {missing_after}")
    
    return df_cleaned

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    
    if df.empty:
        print("Validation failed: DataFrame is empty.")
        return False
    
    if df.shape[0] < min_rows:
        print(f"Validation failed: DataFrame has fewer than {min_rows} rows.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, np.nan, 5],
        'B': [10, np.nan, 10, 30, 40, 50],
        'C': ['x', 'y', 'x', 'y', 'z', np.nan]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    is_valid = validate_data(cleaned_df, required_columns=['A', 'B'], min_rows=3)
    print(f"\nData is valid: {is_valid}")