
import pandas as pd

def clean_dataset(df, column_name):
    """
    Clean a DataFrame by removing duplicate rows and sorting by a specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_name (str): Column name to sort by.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with duplicates removed and sorted.
    """
    if df.empty:
        return df
    
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    df_cleaned = df_cleaned.sort_values(by=column_name).reset_index(drop=True)
    
    return df_cleaned

def validate_data(df, required_columns):
    """
    Validate that the DataFrame contains all required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        bool: True if all required columns are present, False otherwise.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [3, 1, 2, 1, 3, 4],
        'name': ['Charlie', 'Alice', 'Bob', 'Alice', 'Charlie', 'David'],
        'value': [300, 100, 200, 100, 300, 400]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = clean_dataset(df, 'id')
    print("Cleaned DataFrame (sorted by 'id'):")
    print(cleaned_df)
    print()
    
    is_valid = validate_data(cleaned_df, ['id', 'name', 'value'])
    print(f"Data validation passed: {is_valid}")