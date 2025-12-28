
import pandas as pd
import re

def clean_text_column(df, column_name):
    """
    Standardize text by converting to lowercase, removing extra spaces,
    and stripping special characters except alphanumeric and basic punctuation.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str).str.lower()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x))
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^\w\s.,!?-]', '', x))
    df[column_name] = df[column_name].str.strip()
    
    return df

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def clean_dataset(df, text_columns=None, deduplicate=True, subset=None):
    """
    Main function to clean dataset by processing text columns and removing duplicates.
    """
    if text_columns:
        for col in text_columns:
            df = clean_text_column(df, col)
    
    if deduplicate:
        df = remove_duplicates(df, subset=subset)
    
    return df

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['  JOHN DOE  ', 'Jane Smith', 'JOHN DOE', 'Bob@Wilson', 'Alice'],
        'email': ['test@mail.com', 'test@mail.com', 'other@mail.com', 'bob@mail.com', 'alice@mail.com']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, text_columns=['name'], deduplicate=True, subset=['email'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)