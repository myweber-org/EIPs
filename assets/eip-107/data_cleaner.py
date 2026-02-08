
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, convert_types=True):
    """
    Clean a pandas DataFrame by removing duplicates and converting data types.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if convert_types:
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                try:
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col])
                    print(f"Converted column '{col}' to datetime")
                except (ValueError, TypeError):
                    try:
                        cleaned_df[col] = pd.to_numeric(cleaned_df[col])
                        print(f"Converted column '{col}' to numeric")
                    except (ValueError, TypeError):
                        pass
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def validate_data(df, required_columns=None, allow_nan=True):
    """
    Validate DataFrame structure and content.
    """
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    if not allow_nan:
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            print(f"Warning: Found {nan_count} NaN values in the dataset")
    
    return True

def sample_data(df, n=5, random_state=42):
    """
    Return a random sample from the DataFrame.
    """
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=random_state)