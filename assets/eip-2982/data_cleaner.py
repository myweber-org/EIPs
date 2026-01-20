import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
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

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to clean.
    
    Returns:
    pd.DataFrame: DataFrame with cleaned numeric columns.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def validate_dataframe(df, required_columns):
    """
    Validate DataFrame contains required columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if all required columns are present.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    
    return True

def process_dataframe(df, cleaning_config):
    """
    Main function to process DataFrame with cleaning operations.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    cleaning_config (dict): Configuration for cleaning operations.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    if not validate_dataframe(df, cleaning_config.get('required_columns', [])):
        raise ValueError("DataFrame missing required columns")
    
    if cleaning_config.get('remove_duplicates', False):
        df = remove_duplicates(
            df, 
            subset=cleaning_config.get('duplicate_subset'),
            keep=cleaning_config.get('duplicate_keep', 'first')
        )
    
    if cleaning_config.get('clean_numeric', False):
        df = clean_numeric_columns(
            df, 
            cleaning_config.get('numeric_columns', [])
        )
    
    return df