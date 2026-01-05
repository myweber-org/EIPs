
import pandas as pd

def clean_dataset(df):
    """
    Remove duplicate rows and normalize column names.
    """
    # Remove duplicates
    df_cleaned = df.drop_duplicates()
    
    # Normalize column names: strip whitespace, lowercase, replace spaces with underscores
    df_cleaned.columns = (
        df_cleaned.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
    )
    
    return df_cleaned

def validate_data(df, required_columns):
    """
    Check if required columns exist in the DataFrame.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True