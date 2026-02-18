import pandas as pd

def clean_dataset(df, column_names):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing specified string columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_names (list): List of column names to normalize.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Normalize string columns: strip whitespace and convert to lowercase
    for col in column_names:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip().str.lower()
    
    return df_cleaned

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'name': ['  Alice  ', 'Bob', 'alice', 'Bob ', 'Charlie'],
        'age': [25, 30, 25, 30, 35],
        'city': ['  New York', 'London ', 'new york', 'london', 'Paris']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataset(df, ['name', 'city'])
    print(cleaned_df)