
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_clean (list, optional): List of column names to apply string normalization.
            If None, all object dtype columns are cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    removed_duplicates = initial_rows - len(cleaned_df)
    
    if removed_duplicates > 0:
        print(f"Removed {removed_duplicates} duplicate rows.")
    
    # Determine which columns to clean
    if columns_to_clean is None:
        columns_to_clean = cleaned_df.select_dtypes(include=['object']).columns.tolist()
    
    # Normalize string columns
    for column in columns_to_clean:
        if column in cleaned_df.columns and cleaned_df[column].dtype == 'object':
            cleaned_df[column] = cleaned_df[column].apply(_normalize_string)
            print(f"Normalized strings in column: {column}")
    
    return cleaned_df

def _normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters from the edges.
    
    Args:
        text (str): Input string to normalize.
    
    Returns:
        str: Normalized string, or original value if not a string.
    """
    if not isinstance(text, str):
        return text
    
    # Convert to lowercase
    normalized = text.lower()
    
    # Remove leading/trailing whitespace
    normalized = normalized.strip()
    
    # Replace multiple spaces with single space
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of the column containing email addresses.
    
    Returns:
        pd.DataFrame: DataFrame with an additional 'email_valid' boolean column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame.")
    
    validated_df = df.copy()
    
    # Simple email validation regex
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    validated_df['email_valid'] = validated_df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )
    
    valid_count = validated_df['email_valid'].sum()
    total_count = len(validated_df)
    
    print(f"Email validation: {valid_count} valid out of {total_count} ({valid_count/total_count*100:.1f}%)")
    
    return validated_df

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'name': ['John Doe', 'Jane Smith', 'john doe', 'Jane SMITH', 'Bob Johnson'],
#         'email': ['john@example.com', 'jane@example.com', 'invalid-email', 'jane@example.com', 'bob@test.org'],
#         'age': [25, 30, 25, 30, 35]
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     print("\n")
#     
#     # Clean the data
#     cleaned = clean_dataframe(df)
#     print("Cleaned DataFrame:")
#     print(cleaned)
#     print("\n")
#     
#     # Validate emails
#     validated = validate_email_column(cleaned, 'email')
#     print("DataFrame with email validation:")
#     print(validated)