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

def validate_dataframe(df):
    """
    Basic validation of DataFrame structure.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
    
    Returns:
        bool: True if valid, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    if df.shape[0] == 0:
        return False
    return True

def clean_numeric_column(df, column_name):
    """
    Clean a numeric column by removing non-numeric values.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of column to clean.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned column.
    """
    if column_name not in df.columns:
        return df
    
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    return df

def main():
    """
    Example usage of data cleaning functions.
    """
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10, 20, 20, 30, 40],
        'category': ['A', 'B', 'B', 'C', 'D']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = remove_duplicates(df, subset=['id'])
    print("\nDataFrame after removing duplicate IDs:")
    print(cleaned_df)
    
    cleaned_df = clean_numeric_column(cleaned_df, 'value')
    print("\nDataFrame with cleaned numeric column:")
    print(cleaned_df)
    
    is_valid = validate_dataframe(cleaned_df)
    print(f"\nDataFrame is valid: {is_valid}")

if __name__ == "__main__":
    main()