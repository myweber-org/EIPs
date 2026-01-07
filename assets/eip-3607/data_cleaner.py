import pandas as pd

def clean_dataset(df, column_name):
    """
    Clean a specific column in a pandas DataFrame.
    Removes duplicates, strips whitespace, and converts to lowercase.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    # Remove duplicate rows based on the specified column
    df_cleaned = df.drop_duplicates(subset=[column_name])

    # Normalize the string values in the specified column
    df_cleaned[column_name] = df_cleaned[column_name].astype(str).str.strip().str.lower()

    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)

    return df_cleaned

def validate_email_column(df, email_column):
    """
    Validate email addresses in a DataFrame column using a simple regex pattern.
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")

    df['is_valid_email'] = df[email_column].astype(str).apply(lambda x: bool(re.match(pattern, x)))
    return df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'name': ['Alice', 'Bob', 'Alice', 'Charlie', 'bob  '],
        'email': ['alice@example.com', 'bob@test.org', 'alice@example.com', 'invalid-email', 'BOB@TEST.ORG']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)

    cleaned_df = clean_dataset(df, 'name')
    print("\nDataFrame after cleaning 'name' column:")
    print(cleaned_df)

    validated_df = validate_email_column(cleaned_df, 'email')
    print("\nDataFrame with email validation:")
    print(validated_df[['name', 'email', 'is_valid_email']])