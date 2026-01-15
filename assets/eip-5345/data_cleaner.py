
import pandas as pd
import re

def clean_dataset(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df: Input pandas DataFrame
        column_mapping: Dictionary mapping original column names to new names
        drop_duplicates: Whether to remove duplicate rows
        normalize_text: Whether to normalize text columns (strip, lower case)
    
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
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    # Normalize text columns
    if normalize_text:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
            # Remove extra whitespace
            cleaned_df[col] = cleaned_df[col].apply(lambda x: re.sub(r'\s+', ' ', x))
        print(f"Normalized {len(text_columns)} text columns")
    
    # Convert date columns if detected
    date_patterns = ['date', 'time', 'created', 'updated']
    for col in cleaned_df.columns:
        if any(pattern in col.lower() for pattern in date_patterns):
            try:
                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
            except:
                pass
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the cleaned DataFrame meets basic requirements.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        min_rows: Minimum number of rows required
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(df) < min_rows:
        return False, f"Dataset has only {len(df)} rows, minimum required is {min_rows}"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    # Check for excessive null values
    null_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns))
    if null_percentage > 0.5:
        return False, f"High percentage of null values: {null_percentage:.1%}"
    
    return True, "Data validation passed"

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'user_id': [1, 2, 2, 3, 4],
        'user_name': [' John ', 'Alice', 'alice', 'Bob ', ' Charlie '],
        'signup_date': ['2023-01-01', '2023-01-02', '2023-01-02', 'invalid', '2023-01-03'],
        'email': ['JOHN@EXAMPLE.COM', 'alice@example.com', 'alice@example.com', 'bob@example.com', 'charlie@example.com']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned = clean_dataset(df, column_mapping={'user_name': 'username'})
    print("\nCleaned data:")
    print(cleaned)
    
    # Validate
    is_valid, message = validate_data(cleaned, required_columns=['user_id', 'username'])
    print(f"\nValidation: {message}")