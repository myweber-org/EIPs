import pandas as pd
import re

def clean_dataframe(df, text_column):
    """
    Remove duplicate rows and standardize text in specified column.
    """
    # Remove duplicates
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    # Standardize text: lowercase, remove extra whitespace
    if text_column in df_clean.columns:
        df_clean[text_column] = df_clean[text_column].apply(
            lambda x: re.sub(r'\s+', ' ', str(x).strip().lower())
        )
    
    return df_clean

def validate_email(email):
    """
    Simple email validation using regex.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))