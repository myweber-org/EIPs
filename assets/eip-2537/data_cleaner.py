
import pandas as pd

def clean_dataset(df, drop_na=True, rename_columns=True):
    """
    Clean a pandas DataFrame by removing null values and standardizing column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_na (bool): Whether to drop rows with null values. Default is True.
        rename_columns (bool): Whether to rename columns to lowercase with underscores. Default is True.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if drop_na:
        df_clean = df_clean.dropna()
    
    if rename_columns:
        df_clean.columns = (
            df_clean.columns
            .str.lower()
            .str.replace(' ', '_')
            .str.replace(r'[^\w_]', '', regex=True)
        )
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names. Default is None.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"