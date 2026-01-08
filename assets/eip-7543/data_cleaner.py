
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df: pandas DataFrame to clean
        column_mapping: dictionary mapping old column names to new ones
        drop_duplicates: whether to remove duplicate rows
        normalize_text: whether to normalize text columns (strip, lower case)
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    # Rename columns if mapping provided
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    # Remove duplicate rows
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    # Normalize text columns
    if normalize_text:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
            # Remove extra whitespace
            cleaned_df[col] = cleaned_df[col].apply(lambda x: re.sub(r'\s+', ' ', x))
    
    # Convert date columns if detected
    date_pattern = re.compile(r'.*date.*|.*time.*', re.IGNORECASE)
    date_columns = [col for col in cleaned_df.columns if date_pattern.match(col)]
    
    for col in date_columns:
        try:
            cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
        except:
            pass
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of column names that must be present
        min_rows: minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned DataFrame to file.
    
    Args:
        df: pandas DataFrame to save
        output_path: path to save the file
        format: output format ('csv', 'excel', 'parquet')
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'parquet':
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Cleaned data saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'user_id': [1, 2, 2, 3, 4],
        'user_name': [' John Doe ', 'Jane SMITH', 'Jane Smith', 'Bob', 'Alice'],
        'signup_date': ['2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03', '2023-01-04'],
        'email': ['JOHN@EXAMPLE.COM', 'JANE@EXAMPLE.COM', 'jane@example.com', 'bob@example.com', 'alice@example.com']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataframe(df)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the data
    is_valid, message = validate_dataframe(cleaned_df, required_columns=['user_id', 'user_name'])
    print(f"\nValidation: {message}")