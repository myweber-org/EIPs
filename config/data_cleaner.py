import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_column(df, column_name):
    """
    Clean a numeric column by removing non-numeric characters.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of column to clean.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    return df

def validate_email_format(df, email_column):
    """
    Validate email addresses in a column using basic regex pattern.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of email column.
    
    Returns:
        pd.DataFrame: DataFrame with validation results.
    """
    import re
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    df['email_valid'] = df[email_column].apply(
        lambda x: bool(re.match(pattern, str(x))) if pd.notnull(x) else False
    )
    
    return df

def get_cleaning_summary(df_before, df_after):
    """
    Generate a summary of cleaning operations performed.
    
    Args:
        df_before (pd.DataFrame): Original DataFrame.
        df_after (pd.DataFrame): Cleaned DataFrame.
    
    Returns:
        dict: Summary statistics.
    """
    summary = {
        'original_rows': len(df_before),
        'cleaned_rows': len(df_after),
        'rows_removed': len(df_before) - len(df_after),
        'columns_before': list(df_before.columns),
        'columns_after': list(df_after.columns)
    }
    
    return summary