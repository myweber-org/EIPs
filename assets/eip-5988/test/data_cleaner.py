
import pandas as pd
import re

def clean_dataframe(df, columns_to_normalize=None, remove_duplicates=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_normalize (list, optional): List of column names to normalize.
            If None, all object dtype columns are normalized.
        remove_duplicates (bool): Whether to remove duplicate rows.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if columns_to_normalize is None:
        columns_to_normalize = cleaned_df.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns_to_normalize:
        if col in cleaned_df.columns and cleaned_df[col].dtype == 'object':
            cleaned_df[col] = cleaned_df[col].apply(_normalize_string)
            print(f"Normalized column: {col}")
    
    return cleaned_df

def _normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    
    Args:
        text (str): Input string.
    
    Returns:
        str: Normalized string.
    """
    if not isinstance(text, str):
        return text
    
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'[^\w\s-]', '', text)
    
    return text

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of the column containing email addresses.
    
    Returns:
        pd.DataFrame: DataFrame with valid emails and a new 'email_valid' column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame.")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )
    
    valid_count = df['email_valid'].sum()
    print(f"Found {valid_count} valid email addresses out of {len(df)} rows.")
    
    return df
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
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_missing_values(df, strategy='drop', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        strategy (str): 'drop' or 'fill'.
        columns (list, optional): Specific columns to clean.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if strategy == 'drop':
        if columns:
            df = df.dropna(subset=columns)
        else:
            df = df.dropna()
    elif strategy == 'fill':
        if columns:
            for col in columns:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].mean())
        else:
            df = df.fillna(df.mean())
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        bool: True if validation passes.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return False
    
    if df.empty:
        print("DataFrame is empty")
        return False
    
    return True

def process_dataframe(df, operations=None):
    """
    Apply multiple cleaning operations to DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        operations (list): List of operation dictionaries.
    
    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    if operations is None:
        operations = []
    
    for op in operations:
        if op.get('type') == 'remove_duplicates':
            df = remove_duplicates(df, **op.get('params', {}))
        elif op.get('type') == 'clean_missing':
            df = clean_missing_values(df, **op.get('params', {}))
    
    return df