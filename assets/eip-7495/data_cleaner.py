
import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Args:
        dataframe: Input DataFrame
        subset: Column label or sequence of labels to consider for duplicates
        keep: Determines which duplicates to keep ('first', 'last', False)
    
    Returns:
        DataFrame with duplicates removed
    """
    if dataframe.empty:
        return dataframe
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(dataframe) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate row(s)")
    
    return cleaned_df

def clean_numeric_columns(dataframe, columns):
    """
    Clean numeric columns by removing non-numeric characters and converting to float.
    
    Args:
        dataframe: Input DataFrame
        columns: List of column names to clean
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    df = dataframe.copy()
    
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
    
    return df

def validate_dataframe(dataframe, required_columns):
    """
    Validate that DataFrame contains all required columns.
    
    Args:
        dataframe: Input DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        Tuple of (is_valid, missing_columns)
    """
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False, missing_columns
    
    return True, []