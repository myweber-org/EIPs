
import pandas as pd
import numpy as np

def clean_dataset(df, drop_na=True, rename_cols=True):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_na (bool): If True, drop rows with any missing values. Default True.
    rename_cols (bool): If True, rename columns to lowercase with underscores. Default True.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if drop_na:
        df_clean = df_clean.dropna()
    
    if rename_cols:
        df_clean.columns = (
            df_clean.columns
            .str.lower()
            .str.replace(r'[^\w\s]', '', regex=True)
            .str.replace(r'\s+', '_', regex=True)
        )
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'Product Name': ['A', 'B', None, 'D'],
        'Price ($)': [100, 200, 300, None],
        'Quantity in Stock': [10, 20, 30, 40]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataset(df)
    print(cleaned_df)
    
    is_valid = validate_dataframe(cleaned_df, ['product_name', 'price', 'quantity_in_stock'])
    print(f"\nDataFrame validation: {is_valid}")