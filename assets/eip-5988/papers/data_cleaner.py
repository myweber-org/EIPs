def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd

def clean_dataset(df, id_column='id'):
    """
    Clean a pandas DataFrame by removing duplicate rows based on ID column
    and standardizing column names to lowercase with underscores.
    """
    if df.empty:
        return df
    
    # Remove duplicates based on specified ID column
    if id_column in df.columns:
        df_cleaned = df.drop_duplicates(subset=[id_column], keep='first')
    else:
        df_cleaned = df.copy()
    
    # Standardize column names
    df_cleaned.columns = (
        df_cleaned.columns
        .str.lower()
        .str.replace(' ', '_')
        .str.replace(r'[^\w_]', '', regex=True)
    )
    
    return df_cleaned.reset_index(drop=True)

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'ID': [1, 2, 2, 3, 4],
        'First Name': ['John', 'Jane', 'Jane', 'Bob', 'Alice'],
        'Last Name': ['Doe', 'Smith', 'Smith', 'Johnson', 'Brown'],
        'Age': [25, 30, 30, 35, 28]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataset(df, 'ID')
    print(cleaned_df)