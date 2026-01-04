
import pandas as pd

def clean_dataset(df, column_names):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing specified string columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_names (list): List of column names to normalize (strip whitespace, lowercase).
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Normalize string columns
    for col in column_names:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns):
    """
    Validate that the DataFrame contains all required columns and has no empty values.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        tuple: (bool, str) indicating validation success and message.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    empty_rows = df[required_columns].isnull().any(axis=1)
    if empty_rows.any():
        return False, f"Found {empty_rows.sum()} rows with empty values in required columns"
    
    return True, "Data validation passed"

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'name': ['  John  ', 'Jane', 'JOHN', 'Jane', 'Bob'],
        'email': ['john@email.com', 'jane@email.com', 'john@email.com', 'JANE@EMAIL.COM', 'bob@email.com'],
        'age': [25, 30, 25, 30, 35]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned = clean_dataset(df, ['name', 'email'])
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n")
    
    is_valid, message = validate_data(cleaned, ['name', 'email', 'age'])
    print(f"Validation: {is_valid}")
    print(f"Message: {message}")