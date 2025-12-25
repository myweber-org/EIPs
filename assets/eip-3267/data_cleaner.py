
import pandas as pd

def clean_dataset(df, column_names):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing specified string columns.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Normalize string columns: strip whitespace and convert to lowercase
    for col in column_names:
        if col in df_cleaned.columns and df_cleaned[col].dtype == 'object':
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip().str.lower()
    
    return df_cleaned

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column using a simple regex pattern.
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    # Create a mask for valid emails
    valid_mask = df[email_column].astype(str).str.match(pattern)
    
    # Return DataFrame with valid emails and validation status column
    result_df = df.copy()
    result_df['email_valid'] = valid_mask
    
    return result_df[valid_mask], result_df