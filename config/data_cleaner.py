
import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Clean dataset by removing duplicates and handling missing values.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Handle missing values
    # For numerical columns, fill with median
    numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_cleaned[col].isnull().any():
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
    
    # For categorical columns, fill with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_cleaned[col].isnull().any():
            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
    
    return df_cleaned

def validate_data(df):
    """
    Validate data after cleaning.
    """
    # Check for remaining missing values
    missing_values = df.isnull().sum().sum()
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    
    validation_report = {
        'missing_values': missing_values,
        'duplicates': duplicates,
        'total_rows': len(df),
        'total_columns': len(df.columns)
    }
    
    return validation_report

if __name__ == "__main__":
    # Example usage
    data = pd.DataFrame({
        'A': [1, 2, 2, 3, None],
        'B': ['x', 'y', 'y', None, 'z'],
        'C': [1.1, 2.2, 2.2, 3.3, 4.4]
    })
    
    print("Original data:")
    print(data)
    
    cleaned_data = clean_dataset(data)
    print("\nCleaned data:")
    print(cleaned_data)
    
    report = validate_data(cleaned_data)
    print("\nValidation report:")
    for key, value in report.items():
        print(f"{key}: {value}")import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str or dict): Strategy to fill missing values. 
            Can be 'mean', 'median', 'mode', or a dictionary of column:value pairs.
            If None, missing values are not filled.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate a DataFrame for required columns and minimum row count.
    
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
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"