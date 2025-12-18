
import pandas as pd
import re

def clean_dataframe(df, column_names=None):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    """
    df_clean = df.copy()
    
    # Remove duplicate rows
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    removed_duplicates = initial_rows - df_clean.shape[0]
    
    # If specific columns are provided, clean only those
    # Otherwise, clean all object/string columns
    if column_names is None:
        column_names = df_clean.select_dtypes(include=['object']).columns
    
    for col in column_names:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].apply(_normalize_string)
    
    return df_clean, removed_duplicates

def _normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    """
    if pd.isna(text):
        return text
    
    # Convert to string if not already
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters (keep alphanumeric and basic punctuation)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text

def validate_email_column(df, email_column):
    """
    Validate email addresses in a DataFrame column.
    Returns a DataFrame with valid emails and count of invalid entries.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    df_valid = df[df[email_column].str.match(email_pattern, na=False)]
    invalid_count = df.shape[0] - df_valid.shape[0]
    
    return df_valid, invalid_count

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'name': ['John Doe', 'Jane Smith', 'John Doe', 'Alice Johnson  '],
#         'email': ['john@example.com', 'invalid-email', 'john@example.com', 'alice@company.org'],
#         'age': [25, 30, 25, 35]
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     
#     cleaned_df, duplicates_removed = clean_dataframe(df)
#     print(f"\nRemoved {duplicates_removed} duplicate(s)")
#     print("\nCleaned DataFrame:")
#     print(cleaned_df)
#     
#     valid_emails, invalid_count = validate_email_column(cleaned_df, 'email')
#     print(f"\nFound {invalid_count} invalid email(s)")
#     print("\nDataFrame with valid emails:")
#     print(valid_emails)